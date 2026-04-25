/**
 * OpenAI-compatible streaming adapter for local vLLM (Qwen 3.6).
 *
 * Translates Anthropic-format messages/tools → OpenAI /v1/chat/completions,
 * streams the SSE response, and yields AssistantMessage / StreamEvent /
 * SystemAPIErrorMessage values matching the shape expected by QueryEngine.
 *
 * Wire differences handled here:
 *   Anthropic tool_use block   → OpenAI tool_calls array on assistant message
 *   Anthropic tool_result block → OpenAI {role:"tool", tool_call_id, content} message
 *   Anthropic text block       → OpenAI content string / delta.content
 *   Anthropic tools            → OpenAI functions (type:"function", function:{name,description,parameters})
 *
 * The default local deployment alias is the promoted CTF model
 * `qwen3.6-27b-ctf`.
 */

import type {
  BetaContentBlock,
  BetaToolUnion,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import { randomUUID } from 'crypto'
import type { Tool, Tools } from '../../Tool.js'
import type {
  AssistantMessage,
  Message,
  StreamEvent,
  SystemAPIErrorMessage,
} from '../../types/message.js'
import { toolToAPISchema } from '../../utils/api.js'
import {
  createAssistantMessage,
  normalizeMessagesForAPI,
} from '../../utils/messages.js'
import { getLocalApiKey, getLocalModelId, LOCAL_BASE_URL, LOCAL_DEFAULT_MODEL, modelSupportsToolChoice } from '../../utils/model/local.js'
import type { SystemPrompt } from '../../utils/systemPromptType.js'
import type { Options } from './claude.js'

// ---------------------------------------------------------------------------
// Gemma4 native tool-call format recovery
// ---------------------------------------------------------------------------
//
// vLLM's gemma4 tool parser only disables `skip_special_tokens` when the
// request carries a non-empty `tools` list (see
// vllm/tool_parsers/gemma4_tool_parser.py:adjust_request). Fabric makes many
// legitimate side-calls with tools=[] (sub-agent generation, away summary,
// plan-mode subroutines, bash-progress). On those calls vLLM strips the
// `<|tool_call>` markers before the parser can run, and the model's native
// serialization leaks into `content` as raw text:
//
//     <|tool_call>call:Bash{command:<|"|>echo hi<|"|>}<tool_call|>
//
// The functions below mirror gemma4_tool_parser._parse_gemma4_args and let us
// recover those leaked tool calls into proper tool_use blocks. This is a
// defense-in-depth layer; vLLM's parser remains the primary path.

// Mirrors vllm/tool_parsers/gemma4_tool_parser.py:tool_call_regex. Non-greedy
// `[\s\S]*?` finds the first `}<tool_call|>` after the opening `{` — since
// `<tool_call|>` is a unique sentinel, this correctly closes the args even
// when those args contain arbitrarily nested `{...}` objects.
const GEMMA_TOOL_CALL_REGEX =
  /<\|tool_call>call:([\w\-.]+)\{([\s\S]*?)\}<tool_call\|>/g
const GEMMA_STRING_DELIM = '<|"|>'

function parseGemmaValue(raw: string): unknown {
  const v = raw.trim()
  if (!v) return v
  if (v === 'true') return true
  if (v === 'false') return false
  if (v === 'null') return null
  if (/^-?\d+$/.test(v)) return parseInt(v, 10)
  if (/^-?\d*\.\d+$/.test(v)) return parseFloat(v)
  return v
}

function parseGemmaArray(arr: string): unknown[] {
  const out: unknown[] = []
  let i = 0
  const n = arr.length
  while (i < n) {
    while (i < n && (arr[i] === ' ' || arr[i] === ',' || arr[i] === '\n' || arr[i] === '\t')) i++
    if (i >= n) break
    if (arr.startsWith(GEMMA_STRING_DELIM, i)) {
      i += GEMMA_STRING_DELIM.length
      const end = arr.indexOf(GEMMA_STRING_DELIM, i)
      if (end === -1) { out.push(arr.slice(i)); break }
      out.push(arr.slice(i, end))
      i = end + GEMMA_STRING_DELIM.length
    } else if (arr[i] === '{') {
      let depth = 1; const start = i + 1; i++
      while (i < n && depth > 0) {
        if (arr.startsWith(GEMMA_STRING_DELIM, i)) {
          i += GEMMA_STRING_DELIM.length
          const end = arr.indexOf(GEMMA_STRING_DELIM, i)
          i = end === -1 ? n : end + GEMMA_STRING_DELIM.length
          continue
        }
        if (arr[i] === '{') depth++
        else if (arr[i] === '}') depth--
        i++
      }
      out.push(parseGemmaArgs(arr.slice(start, i - 1)))
    } else {
      const start = i
      while (i < n && arr[i] !== ',' && arr[i] !== ']') i++
      out.push(parseGemmaValue(arr.slice(start, i)))
    }
  }
  return out
}

function parseGemmaArgs(args: string): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  if (!args || !args.trim()) return result
  let i = 0
  const n = args.length
  while (i < n) {
    while (i < n && (args[i] === ' ' || args[i] === ',' || args[i] === '\n' || args[i] === '\t')) i++
    if (i >= n) break
    const keyStart = i
    while (i < n && args[i] !== ':') i++
    if (i >= n) break
    const key = args.slice(keyStart, i).trim()
    i++ // skip ':'
    while (i < n && (args[i] === ' ' || args[i] === '\n' || args[i] === '\t')) i++
    if (i >= n) { result[key] = ''; break }
    if (args.startsWith(GEMMA_STRING_DELIM, i)) {
      i += GEMMA_STRING_DELIM.length
      const end = args.indexOf(GEMMA_STRING_DELIM, i)
      if (end === -1) { result[key] = args.slice(i); break }
      result[key] = args.slice(i, end)
      i = end + GEMMA_STRING_DELIM.length
    } else if (args[i] === '{') {
      let depth = 1; const start = i + 1; i++
      while (i < n && depth > 0) {
        if (args.startsWith(GEMMA_STRING_DELIM, i)) {
          i += GEMMA_STRING_DELIM.length
          const end = args.indexOf(GEMMA_STRING_DELIM, i)
          i = end === -1 ? n : end + GEMMA_STRING_DELIM.length
          continue
        }
        if (args[i] === '{') depth++
        else if (args[i] === '}') depth--
        i++
      }
      result[key] = parseGemmaArgs(args.slice(start, i - 1))
    } else if (args[i] === '[') {
      let depth = 1; const start = i + 1; i++
      while (i < n && depth > 0) {
        if (args.startsWith(GEMMA_STRING_DELIM, i)) {
          i += GEMMA_STRING_DELIM.length
          const end = args.indexOf(GEMMA_STRING_DELIM, i)
          i = end === -1 ? n : end + GEMMA_STRING_DELIM.length
          continue
        }
        if (args[i] === '[') depth++
        else if (args[i] === ']') depth--
        i++
      }
      result[key] = parseGemmaArray(args.slice(start, i - 1))
    } else {
      const start = i
      while (i < n && args[i] !== ',' && args[i] !== '}' && args[i] !== ']') i++
      result[key] = parseGemmaValue(args.slice(start, i))
    }
  }
  return result
}

/**
 * Scan accumulated content text for leaked Gemma4-format tool calls. Returns
 * the cleaned text plus any recovered tool calls.
 */
export function recoverLeakedGemmaToolCalls(text: string): {
  cleanedText: string
  recovered: Array<{ name: string; args: Record<string, unknown> }>
} {
  if (!text.includes('<|tool_call>')) {
    return { cleanedText: text, recovered: [] }
  }
  const recovered: Array<{ name: string; args: Record<string, unknown> }> = []
  const cleanedText = text.replace(
    GEMMA_TOOL_CALL_REGEX,
    (_match: string, name: string, argsStr: string) => {
      try {
        recovered.push({ name, args: parseGemmaArgs(argsStr) })
      } catch {
        recovered.push({ name, args: { _parse_error: argsStr } })
      }
      return ''
    },
  )
  return { cleanedText: cleanedText.trim(), recovered }
}

// ---------------------------------------------------------------------------
// Permissive JSON repair for malformed tool args (Fix 2)
// ---------------------------------------------------------------------------
//
// Local models routinely emit JSON tool arguments with trailing commas,
// unquoted keys, prose preambles, or unbalanced braces. Calling the tool
// with `{raw: "<garbage>"}` guarantees a confusing downstream error; better
// to attempt a targeted repair, and surface a clear error if even that fails
// so the model can self-correct on the next turn.

export function repairToolArgsJson(input: string): Record<string, unknown> | null {
  if (!input || !input.trim()) return {}
  try {
    return JSON.parse(input) as Record<string, unknown>
  } catch {
    // fallthrough to repair
  }
  let s = input.trim()
  // Strip a leading prose preamble before the first '{'.
  const firstBrace = s.indexOf('{')
  if (firstBrace > 0) s = s.slice(firstBrace)
  // Strip trailing prose after the last balanced '}'.
  const lastBrace = s.lastIndexOf('}')
  if (lastBrace !== -1 && lastBrace < s.length - 1) s = s.slice(0, lastBrace + 1)
  // Remove trailing commas before } or ].
  s = s.replace(/,(\s*[}\]])/g, '$1')
  // Balance unclosed braces / brackets — append closers in stack order.
  const stack: string[] = []
  let inString = false; let escape = false
  for (const ch of s) {
    if (escape) { escape = false; continue }
    if (ch === '\\') { escape = true; continue }
    if (ch === '"') { inString = !inString; continue }
    if (inString) continue
    if (ch === '{') stack.push('}')
    else if (ch === '[') stack.push(']')
    else if (ch === '}' || ch === ']') stack.pop()
  }
  while (stack.length) s += stack.pop()
  try {
    return JSON.parse(s) as Record<string, unknown>
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// OpenAI wire types (minimal — only what we send/receive)
// ---------------------------------------------------------------------------

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | null
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
  name?: string
}

interface OpenAIToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

interface OpenAIFunction {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

interface OpenAIDeltaChunk {
  choices: Array<{
    delta: {
      role?: string
      content?: string | null
      // vLLM with --reasoning-parser (glm45 / qwen3) splits the model's
      // chain-of-thought out of `content` into a separate stream. We surface
      // this as a Beta thinking block so the UI can render or collapse it
      // instead of dropping it silently.
      reasoning_content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        type?: string
        function?: {
          name?: string
          arguments?: string
        }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

// ---------------------------------------------------------------------------
// Anthropic → OpenAI message conversion
// ---------------------------------------------------------------------------

function convertMessagesToOpenAI(
  messages: Message[],
): OpenAIMessage[] {
  const out: OpenAIMessage[] = []

  for (const msg of messages) {
    if (msg.type === 'user') {
      const userMsg = msg as { type: 'user'; message: { content: string | Array<{ type: string; text?: string; tool_use_id?: string; content?: unknown }> } }
      const content = userMsg.message.content
      if (typeof content === 'string') {
        out.push({ role: 'user', content })
        continue
      }

      // Separate tool_result blocks into individual tool messages;
      // everything else becomes a single user message.
      const toolResults: Array<{ tool_call_id: string; content: string }> = []
      const textParts: string[] = []

      for (const block of content) {
        if (block.type === 'tool_result') {
          const resultContent = Array.isArray(block.content)
            ? block.content
                .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
                .map(b => b.text)
                .join('\n')
            : typeof block.content === 'string'
              ? block.content
              : ''
          toolResults.push({
            tool_call_id: block.tool_use_id,
            content: resultContent,
          })
        } else if (block.type === 'text') {
          if (block.text) textParts.push(block.text)
        }
        // image, document, etc. — skip for now
      }

      for (const tr of toolResults) {
        out.push({
          role: 'tool',
          tool_call_id: tr.tool_call_id,
          content: tr.content,
        })
      }

      if (textParts.length > 0) {
        out.push({ role: 'user', content: textParts.join('\n') })
      }
    } else {
      // assistant message
      const assistantMsg = msg as AssistantMessage
      const content = assistantMsg.message.content

      if (typeof content === 'string') {
        out.push({ role: 'assistant', content })
        continue
      }

      const textParts: string[] = []
      const toolCalls: OpenAIToolCall[] = []

      for (const block of content) {
        if (block.type === 'text') {
          if (block.text) textParts.push(block.text)
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            type: 'function',
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input),
            },
          })
        }
        // thinking, redacted_thinking — skip
      }

      const oaiMsg: OpenAIMessage = {
        role: 'assistant',
        content: textParts.join('\n') || null,
      }
      if (toolCalls.length > 0) {
        oaiMsg.tool_calls = toolCalls
      }
      out.push(oaiMsg)
    }
  }

  return out
}

async function convertToolsToOpenAI(
  tools: Tools,
  options: Options,
): Promise<OpenAIFunction[]> {
  const result: OpenAIFunction[] = []

  for (const tool of tools) {
    if (typeof tool === 'function') continue
    try {
      const schema: BetaToolUnion = await toolToAPISchema(tool as Tool, {
        getToolPermissionContext: options.getToolPermissionContext,
        tools,
        agents: options.agents,
        allowedAgentTypes: options.allowedAgentTypes,
        model: options.model,
      })
      const fn = schema as { name: string; description?: string; input_schema: Record<string, unknown> }
      result.push({
        type: 'function',
        function: {
          name: fn.name,
          description: fn.description,
          parameters: fn.input_schema,
        },
      })
    } catch {
      // Skip tools that fail schema generation
    }
  }

  return result
}

// ---------------------------------------------------------------------------
// AbortSignal composition helper
// ---------------------------------------------------------------------------

/**
 * Returns a single AbortSignal that fires when any of the input signals
 * fires. Forwards the first-fired signal's reason so the downstream fetch
 * surfaces a meaningful cause (caller cancel vs idle-stream watchdog).
 * If any input is already aborted, returns a pre-aborted signal.
 */
function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController()
  for (const s of signals) {
    if (s.aborted) {
      controller.abort(s.reason)
      return controller.signal
    }
    s.addEventListener(
      'abort',
      () => controller.abort(s.reason),
      { once: true },
    )
  }
  return controller.signal
}

// ---------------------------------------------------------------------------
// Main streaming function
// ---------------------------------------------------------------------------

/**
 * Caller-supplied recovery hook for HTTP 400 context-overflow.
 *
 * When vLLM rejects a request because input + max_tokens exceeds the model's
 * context window, openai-local.ts can only shrink `max_tokens`. That fails
 * when the server's "prompt contains at least N input tokens" undercounts the
 * true input and the retry still overflows. Callers that own the message
 * history (query.ts has `deps.autocompact`) can provide this callback to
 * force a compaction before the retry. Return `null` to skip compaction and
 * retry with the original messages.
 */
export type OnContextOverflow = (limit: {
  contextWindow: number
  inputTokens: number
}) => Promise<{ messages: Message[] } | null>

// Abort the streaming read if vLLM stalls for this long between SSE chunks.
// This catches dead / hung endpoints without penalizing slow-but-progressing
// long generations (each delivered chunk resets the timer).
const STREAM_IDLE_TIMEOUT_MS = 120_000

// Throughput-floor watchdog (Fix 5). The idle timer above only fires on full
// silence; a model trickling tokens at <5/s for many minutes never trips it
// even though the agent is functionally stuck (low-signal reasoning loops,
// repetitive output, etc.). Once we've seen STALL_WARMUP_TOKENS we start
// measuring sustained throughput; if it stays below STALL_MIN_TOKENS_PER_SEC
// over STALL_WINDOW_MS, we abort.
const STALL_WARMUP_TOKENS = 200
const STALL_MIN_TOKENS_PER_SEC = 5
const STALL_WINDOW_MS = 30_000

// Default ceiling on `max_tokens` per backend, sized to fit the served context
// window in /project/inniang/inference/models/*.sh. Keys are matched via
// getLocalModelId() so partial / suffix matches work the same way as elsewhere.
const MAX_OUTPUT_TOKENS_PER_MODEL: Record<string, number> = {
  'qwen3.6-30b-a3b': 32_000,
  'qwen3.6-27b-dense': 32_000,
  'qwen3.6-27b-ctf': 32_000,
  'gemma4-31b': 32_000,
}
const DEFAULT_MAX_OUTPUT_TOKENS = 32_000

export async function* queryModelLocalStreaming({
  messages,
  systemPrompt,
  tools,
  signal,
  options,
  onContextOverflow,
}: {
  messages: Message[]
  systemPrompt: SystemPrompt
  tools: Tools
  signal: AbortSignal
  options: Options
  onContextOverflow?: OnContextOverflow
}): AsyncGenerator<StreamEvent | AssistantMessage | SystemAPIErrorMessage, void> {
  const model = options.model || LOCAL_DEFAULT_MODEL
  const baseUrl = LOCAL_BASE_URL
  const apiKey = getLocalApiKey()

  // Build a composite signal: caller cancellation OR an idle-stream watchdog.
  // The watchdog fires when no bytes arrive for STREAM_IDLE_TIMEOUT_MS.
  const watchdog = new AbortController()
  const compositeSignal = anySignal([signal, watchdog.signal])
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  const armIdleTimer = () => {
    clearIdleTimer()
    idleTimer = setTimeout(() => {
      watchdog.abort(
        new Error(
          `Local vLLM stream idle for ${STREAM_IDLE_TIMEOUT_MS / 1000}s — aborting`,
        ),
      )
    }, STREAM_IDLE_TIMEOUT_MS)
  }
  const clearIdleTimer = () => {
    if (idleTimer !== undefined) {
      clearTimeout(idleTimer)
      idleTimer = undefined
    }
  }

  // Normalize messages to (UserMessage | AssistantMessage)[]
  let mutableMessages = messages
  let normalized = normalizeMessagesForAPI(mutableMessages)

  // System prompt
  const systemText = systemPrompt.join('\n\n')
  const rebuildOAIMessages = (): OpenAIMessage[] => {
    const out: OpenAIMessage[] = []
    if (systemText.trim()) {
      out.push({ role: 'system', content: systemText })
    }
    out.push(...convertMessagesToOpenAI(normalized))
    return out
  }

  // Convert tools
  const oaiFunctions = tools.length > 0 ? await convertToolsToOpenAI(tools, options) : []

  const localId = getLocalModelId(model)
  const perModelCap = localId !== undefined
    ? MAX_OUTPUT_TOKENS_PER_MODEL[localId] ?? DEFAULT_MAX_OUTPUT_TOKENS
    : DEFAULT_MAX_OUTPUT_TOKENS

  const body: Record<string, unknown> = {
    model,
    messages: rebuildOAIMessages(),
    stream: true,
    max_tokens: options.maxOutputTokensOverride ?? perModelCap,
    temperature: options.temperatureOverride ?? 0.6,
  }

  if (oaiFunctions.length > 0) {
    body.tools = oaiFunctions
    if (modelSupportsToolChoice(model)) {
      body.tool_choice = 'auto'
    }
  }

  const sendRequest = async (): Promise<Response> =>
    fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: compositeSignal,
    })

  let response: Response
  try {
    response = await sendRequest()
  } catch (err: unknown) {
    clearIdleTimer()
    const errMsg = err instanceof Error ? err.message : String(err)
    yield createAssistantMessage({
      content: `Failed to connect to local vLLM endpoint (${baseUrl}): ${errMsg}`,
    }) as unknown as SystemAPIErrorMessage
    return
  }

  // On context-overflow (HTTP 400), parse the vLLM error for actual token
  // counts. Two-step recovery: (a) if the caller provided onContextOverflow
  // — forced autocompact of the message history — run it first; (b) shrink
  // max_tokens to fit whatever input remains. Both are independent belts; a
  // compaction alone may still overflow if the server's input count creeps up
  // between turns, and a max_tokens shrink alone fails when the server's
  // "at least N" is an under-estimate.
  //
  // We iterate up to MAX_OVERFLOW_ATTEMPTS times because vLLM's "at least N
  // input tokens" routinely under-reports — the first shrink lands just over
  // the limit, and a second pass with the now-correct count fits. Bounded so
  // a hard impossibility doesn't loop forever.
  const MAX_OVERFLOW_ATTEMPTS = 3
  let overflowAttempts = 0
  let recovered = response.ok
  while (!response.ok && response.status === 400 && overflowAttempts < MAX_OVERFLOW_ATTEMPTS) {
    overflowAttempts += 1
    const errText = await response.text().catch(() => '')
    let parsed = false
    try {
      const errJson = JSON.parse(errText) as { error?: { message?: string } }
      const msg = errJson?.error?.message ?? ''
      const limitMatch = msg.match(/maximum context length is (\d+)/)
      const inputMatch = msg.match(/prompt contains at least (\d+) input tokens/)
      if (limitMatch && inputMatch) {
        parsed = true
        const contextLimit = parseInt(limitMatch[1]!, 10)
        const inputTokens = parseInt(inputMatch[1]!, 10)

        // Compact only on the first attempt — message-history compaction is
        // expensive and converges in one pass; subsequent retries just need a
        // tighter max_tokens shrink.
        if (overflowAttempts === 1 && onContextOverflow) {
          try {
            const compactResult = await onContextOverflow({
              contextWindow: contextLimit,
              inputTokens,
            })
            if (compactResult) {
              mutableMessages = compactResult.messages
              normalized = normalizeMessagesForAPI(mutableMessages)
              body.messages = rebuildOAIMessages()
            }
          } catch {
            // Compaction is best-effort; fall through to max_tokens shrink.
          }
        }

        // Buffer scales with attempt count: 256, 512, 1024 — each retry leaves
        // more room than the last to absorb vLLM's input-count undercount.
        const buffer = 256 << (overflowAttempts - 1)
        const safeMaxTokens = contextLimit - inputTokens - buffer
        if (safeMaxTokens > 0) {
          body.max_tokens = safeMaxTokens
        } else {
          // No room even with full buffer — bail out, history is too big.
          break
        }
        response = await sendRequest()
      }
    } catch {
      // JSON parse failed — error isn't the shape we recognize; stop retrying.
    }
    if (!parsed) break
  }
  recovered = response.ok
  if (response.status === 400 && !recovered) {
    clearIdleTimer()
    const text = await response.text().catch(() => '')
    yield createAssistantMessage({
      content: `Local vLLM request failed (HTTP ${response.status}) after ${overflowAttempts} recovery attempts: ${text}`,
    }) as unknown as AssistantMessage
    return
  }
  if (!response.ok) {
    clearIdleTimer()
    const text = await response.text().catch(() => '')
    let hint = text
    try {
      const errJson = JSON.parse(text) as { error?: { message?: string } }
      const msg = errJson?.error?.message ?? errJson?.error?.type ?? ''
      if (msg) {
        // vLLM returns a descriptive message for unsupported models, connection errors, etc.
        hint = msg
      }
    } catch {
      // text is already a useful hint
    }
    yield createAssistantMessage({
      content: `Local vLLM request failed (HTTP ${response.status}): ${hint}`,
    }) as unknown as AssistantMessage
    return
  }

  // Stream SSE
  const reader = response.body?.getReader()
  if (!reader) {
    clearIdleTimer()
    yield createAssistantMessage({ content: 'No response body from local vLLM' })
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''

  // Accumulated state for the final AssistantMessage
  let accText = ''
  let accReasoning = ''
  const accToolCalls: Map<number, { id: string; name: string; args: string }> = new Map()

  let promptTokens = 0
  let completionTokens = 0
  let streamDone = false

  // Throughput-floor watchdog state (Fix 5). Track when we cross the warmup
  // threshold and the time/token count at that crossing; afterwards, abort if
  // the rolling rate falls below the floor for STALL_WINDOW_MS.
  let throughputBaselineMs = 0
  let throughputBaselineTokens = 0

  try {
    armIdleTimer()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      // Each delivered chunk resets the idle watchdog; the stream is only
      // considered stuck if STREAM_IDLE_TIMEOUT_MS elapses with zero bytes.
      armIdleTimer()

      // Throughput-floor check. We only start measuring after the warmup
      // window because vLLM sometimes emits a flurry of empty/usage-only
      // chunks during prefill that would skew an early measurement.
      if (completionTokens >= STALL_WARMUP_TOKENS) {
        if (throughputBaselineMs === 0) {
          throughputBaselineMs = Date.now()
          throughputBaselineTokens = completionTokens
        } else {
          const elapsed = Date.now() - throughputBaselineMs
          if (elapsed >= STALL_WINDOW_MS) {
            const tokens = completionTokens - throughputBaselineTokens
            const tokensPerSec = (tokens * 1000) / elapsed
            if (tokensPerSec < STALL_MIN_TOKENS_PER_SEC) {
              watchdog.abort(
                new Error(
                  `Local vLLM stream below ${STALL_MIN_TOKENS_PER_SEC} tok/s for ${STALL_WINDOW_MS / 1000}s — aborting (got ${tokensPerSec.toFixed(2)} tok/s)`,
                ),
              )
              break
            }
            // Reset window for the next measurement period.
            throughputBaselineMs = Date.now()
            throughputBaselineTokens = completionTokens
          }
        }
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || !trimmed.startsWith('data:')) continue
        const data = trimmed.slice(5).trim()
        if (data === '[DONE]') {
          streamDone = true
          break
        }

        let chunk: OpenAIDeltaChunk
        try {
          chunk = JSON.parse(data) as OpenAIDeltaChunk
        } catch {
          continue
        }

        if (chunk.usage) {
          promptTokens = chunk.usage.prompt_tokens ?? 0
          completionTokens = chunk.usage.completion_tokens ?? 0
        }

        const choice = chunk.choices?.[0]
        if (!choice) continue

        const delta = choice.delta

        if (delta.content) {
          accText += delta.content
        }

        if (delta.reasoning_content) {
          accReasoning += delta.reasoning_content
        }

        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index ?? 0
            if (!accToolCalls.has(idx)) {
              accToolCalls.set(idx, { id: tc.id ?? '', name: '', args: '' })
            }
            const entry = accToolCalls.get(idx)!
            if (tc.id) entry.id = tc.id
            if (tc.function?.name) entry.name += tc.function.name
            if (tc.function?.arguments) entry.args += tc.function.arguments
          }
        }
      }
      if (streamDone) break
    }
  } finally {
    clearIdleTimer()
    reader.releaseLock()
  }

  // Fix 1: recover any Gemma4-format tool calls that leaked into accText.
  // Mutates accText (strips matched substrings) and appends recovered calls
  // to accToolCalls so they flow through the same tool_use block path below.
  const leakRecovery = recoverLeakedGemmaToolCalls(accText)
  accText = leakRecovery.cleanedText
  for (const recovered of leakRecovery.recovered) {
    const idx = accToolCalls.size
    accToolCalls.set(idx, {
      id: '',
      name: recovered.name,
      args: JSON.stringify(recovered.args),
    })
  }

  // Build the final content blocks. Thinking goes first so the UI renders
  // the model's reasoning above its final answer, matching how Anthropic
  // models stream thinking + text.
  const contentBlocks: BetaContentBlock[] = []

  if (accReasoning) {
    contentBlocks.push({
      type: 'thinking',
      thinking: accReasoning,
      // vLLM doesn't sign reasoning, so we leave signature empty. The SDK
      // requires the field but accepts an empty string for locally-served
      // models that don't participate in Anthropic's signed-thinking flow.
      signature: '',
    })
  }

  if (accText) {
    contentBlocks.push({ type: 'text', text: accText, citations: [] })
  }

  for (const [, tc] of accToolCalls) {
    // Fix 2: try strict JSON.parse, then permissive repair. If both fail,
    // surface a text block describing the failure rather than calling the
    // tool with a meaningless `{raw: "..."}` payload.
    const repaired = repairToolArgsJson(tc.args)
    if (repaired === null) {
      contentBlocks.push({
        type: 'text',
        text: `(model emitted invalid JSON arguments for tool \`${tc.name}\` — please retry with valid JSON. Raw: ${tc.args.slice(0, 200)})`,
        citations: [],
      })
      continue
    }
    contentBlocks.push({
      type: 'tool_use',
      id: tc.id || `toolu_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
      name: tc.name,
      input: repaired,
    })
  }

  // Fix 3: empty-turn nudge. If the model produced tokens but no usable
  // content or tool_calls, surface that explicitly instead of pushing a
  // silent empty text block — otherwise the agent loop spins another turn
  // on nothing.
  if (contentBlocks.length === 0) {
    if (completionTokens > 0) {
      contentBlocks.push({
        type: 'text',
        text: '(model returned an empty response — retry suggested)',
        citations: [],
      })
    } else {
      contentBlocks.push({ type: 'text', text: '', citations: [] })
    }
  }

  const assistantMsg = createAssistantMessage({
    content: contentBlocks,
    model,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    } as Parameters<typeof createAssistantMessage>[0]['usage'],
  })

  yield assistantMsg
}

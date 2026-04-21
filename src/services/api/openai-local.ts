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
import { getLocalApiKey, LOCAL_BASE_URL, LOCAL_DEFAULT_MODEL } from '../../utils/model/local.js'
import type { SystemPrompt } from '../../utils/systemPromptType.js'
import type { Options } from './claude.js'

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
// Main streaming function
// ---------------------------------------------------------------------------

export async function* queryModelLocalStreaming({
  messages,
  systemPrompt,
  tools,
  signal,
  options,
}: {
  messages: Message[]
  systemPrompt: SystemPrompt
  tools: Tools
  signal: AbortSignal
  options: Options
}): AsyncGenerator<StreamEvent | AssistantMessage | SystemAPIErrorMessage, void> {
  const model = options.model || LOCAL_DEFAULT_MODEL
  const baseUrl = LOCAL_BASE_URL
  const apiKey = getLocalApiKey()

  // Normalize messages to (UserMessage | AssistantMessage)[]
  const normalized = normalizeMessagesForAPI(messages)

  // Convert to OpenAI format
  const oaiMessages: OpenAIMessage[] = []

  // System prompt
  const systemText = systemPrompt.join('\n\n')
  if (systemText.trim()) {
    oaiMessages.push({ role: 'system', content: systemText })
  }

  oaiMessages.push(...convertMessagesToOpenAI(normalized))

  // Convert tools
  const oaiFunctions = tools.length > 0 ? await convertToolsToOpenAI(tools, options) : []

  const body: Record<string, unknown> = {
    model,
    messages: oaiMessages,
    stream: true,
    max_tokens: options.maxOutputTokensOverride ?? 32_000,
    temperature: options.temperatureOverride ?? 0.6,
  }

  if (oaiFunctions.length > 0) {
    body.tools = oaiFunctions
    body.tool_choice = 'auto'
  }

  let response: Response
  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err)
    yield createAssistantMessage({
      content: `Failed to connect to local vLLM endpoint (${baseUrl}): ${errMsg}`,
    }) as unknown as SystemAPIErrorMessage
    return
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    yield createAssistantMessage({
      content: `Local vLLM request failed (HTTP ${response.status}): ${text}`,
    }) as unknown as AssistantMessage
    return
  }

  // Stream SSE
  const reader = response.body?.getReader()
  if (!reader) {
    yield createAssistantMessage({ content: 'No response body from local vLLM' })
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''

  // Accumulated state for the final AssistantMessage
  let accText = ''
  const accToolCalls: Map<number, { id: string; name: string; args: string }> = new Map()

  let promptTokens = 0
  let completionTokens = 0

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || !trimmed.startsWith('data:')) continue
        const data = trimmed.slice(5).trim()
        if (data === '[DONE]') break

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
    }
  } finally {
    reader.releaseLock()
  }

  // Build the final content blocks
  const contentBlocks: BetaContentBlock[] = []

  if (accText) {
    contentBlocks.push({ type: 'text', text: accText })
  }

  for (const [, tc] of accToolCalls) {
    let input: Record<string, unknown> = {}
    try {
      input = JSON.parse(tc.args) as Record<string, unknown>
    } catch {
      input = { raw: tc.args }
    }
    contentBlocks.push({
      type: 'tool_use',
      id: tc.id || `toolu_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
      name: tc.name,
      input,
    })
  }

  if (contentBlocks.length === 0) {
    contentBlocks.push({ type: 'text', text: '' })
  }

  const assistantMsg = createAssistantMessage({
    content: contentBlocks,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    } as Parameters<typeof createAssistantMessage>[0]['usage'],
  })

  yield assistantMsg
}

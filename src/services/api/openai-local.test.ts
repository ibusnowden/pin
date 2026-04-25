import { afterEach, describe, expect, it } from 'bun:test'

import { getEmptyToolPermissionContext } from '../../Tool.js'
import { createUserMessage } from '../../utils/messages.js'
import { LOCAL_DEFAULT_MODEL } from '../../utils/model/local.js'
import {
  queryModelLocalStreaming,
  recoverLeakedGemmaToolCalls,
  repairToolArgsJson,
} from './openai-local.js'

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      const enc = new TextEncoder()
      for (const c of chunks) controller.enqueue(enc.encode(c))
      controller.close()
    },
  })
}

function sseEvent(payload: object): string {
  return `data: ${JSON.stringify(payload)}\n\n`
}

const baseQueryArgs = {
  messages: [createUserMessage({ content: 'go' })],
  systemPrompt: ['system'],
  tools: [] as never,
  signal: new AbortController().signal,
  options: {
    model: LOCAL_DEFAULT_MODEL,
    getToolPermissionContext: async () => getEmptyToolPermissionContext(),
    agents: [],
    allowedAgentTypes: [],
    maxOutputTokensOverride: 64,
  } as never,
}

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe('queryModelLocalStreaming', () => {
  it('sends the promoted alias and tool metadata to local vLLM', async () => {
    const requests: Array<Record<string, unknown>> = []
    globalThis.fetch = (async (_input, init) => {
      requests.push(JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>)
      return new Response(
        `data: ${JSON.stringify({ choices: [{ delta: { content: 'ok' } }] })}\n\n` +
          'data: [DONE]\n\n',
        {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        },
      )
    }) as typeof fetch

    const tool = {
      name: 'bash',
      inputSchema: {} as never,
      inputJSONSchema: {
        type: 'object',
        properties: {
          command: { type: 'string' },
        },
      },
      isEnabled: () => true,
      userFacingName: () => 'bash',
      renderToolUseMessage: () => '',
      call: async () => ({ data: '' }),
      description: async () => '',
      prompt: async () => 'Run a shell command',
      isReadOnly: () => false,
      isMcp: false,
      needsPermissions: () => true,
    } as const

    for await (const _chunk of queryModelLocalStreaming({
      messages: [createUserMessage({ content: 'inspect the workspace' })],
      systemPrompt: ['You are a local bug-finding agent.'],
      tools: [tool as never],
      signal: new AbortController().signal,
      options: {
        model: LOCAL_DEFAULT_MODEL,
        getToolPermissionContext: async () => getEmptyToolPermissionContext(),
        agents: [],
        allowedAgentTypes: [],
        maxOutputTokensOverride: 256,
      } as never,
    })) {
      // drain the stream
    }

    expect(requests).toHaveLength(1)
    expect(requests[0]?.model).toBe(LOCAL_DEFAULT_MODEL)
    expect(requests[0]?.tool_choice).toBe('auto')
    expect(Array.isArray(requests[0]?.tools)).toBe(true)
    expect((requests[0]?.tools as Array<Record<string, unknown>>)[0]?.function).toMatchObject({
      name: 'bash',
    })
  })

  it('stops reading when the local SSE stream sends DONE', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              new TextEncoder().encode(
                `data: ${JSON.stringify({ choices: [{ delta: { content: 'ok' } }] })}\n\n` +
                  'data: [DONE]\n\n',
              ),
            )
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        },
      )
    }) as typeof fetch

    const chunks = []
    for await (const chunk of queryModelLocalStreaming({
      messages: [createUserMessage({ content: 'inspect the workspace' })],
      systemPrompt: ['You are a local bug-finding agent.'],
      tools: [],
      signal: new AbortController().signal,
      options: {
        model: LOCAL_DEFAULT_MODEL,
        getToolPermissionContext: async () => getEmptyToolPermissionContext(),
        agents: [],
        allowedAgentTypes: [],
        maxOutputTokensOverride: 256,
      } as never,
    })) {
      chunks.push(chunk)
    }

    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.type).toBe('assistant')
  })

  // ---------------------------------------------------------------------
  // Fix 1: recover leaked Gemma4 tool-call format from accText
  // ---------------------------------------------------------------------
  it('recovers a Gemma4-format tool call leaked into content text', async () => {
    const leaked = '<|tool_call>call:Bash{command:<|"|>echo hi<|"|>}<tool_call|>'
    globalThis.fetch = (async () => {
      return new Response(
        sseStream([
          sseEvent({ choices: [{ delta: { content: leaked } }] }),
          'data: [DONE]\n\n',
        ]),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )
    }) as typeof fetch

    const chunks: unknown[] = []
    for await (const chunk of queryModelLocalStreaming(baseQueryArgs)) {
      chunks.push(chunk)
    }

    expect(chunks).toHaveLength(1)
    const msg = chunks[0] as {
      type: string
      message: { content: Array<Record<string, unknown>> }
    }
    expect(msg.type).toBe('assistant')
    const blocks = msg.message.content
    const toolUse = blocks.find(b => b.type === 'tool_use') as
      | { type: 'tool_use'; name: string; input: Record<string, unknown> }
      | undefined
    expect(toolUse).toBeDefined()
    expect(toolUse?.name).toBe('Bash')
    expect(toolUse?.input).toEqual({ command: 'echo hi' })
    const text = blocks.find(b => b.type === 'text') as
      | { type: 'text'; text: string }
      | undefined
    expect(text?.text ?? '').not.toContain('<|tool_call>')
  })

  // ---------------------------------------------------------------------
  // Fix 2: malformed JSON tool-call arguments
  // ---------------------------------------------------------------------
  it('repairs a tool_call with a missing closing brace', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        sseStream([
          sseEvent({
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: 'tc1',
                      function: { name: 'bash', arguments: '{"command": "ls"' },
                    },
                  ],
                },
              },
            ],
          }),
          'data: [DONE]\n\n',
        ]),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )
    }) as typeof fetch

    const chunks: unknown[] = []
    for await (const chunk of queryModelLocalStreaming(baseQueryArgs)) {
      chunks.push(chunk)
    }
    const msg = chunks[0] as {
      message: { content: Array<Record<string, unknown>> }
    }
    const toolUse = msg.message.content.find(b => b.type === 'tool_use') as
      | { input: Record<string, unknown> }
      | undefined
    expect(toolUse?.input).toEqual({ command: 'ls' })
  })

  it('surfaces a text block instead of calling a tool with garbage args', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        sseStream([
          sseEvent({
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: 'tc1',
                      function: { name: 'bash', arguments: '!@#$%^&*' },
                    },
                  ],
                },
              },
            ],
          }),
          'data: [DONE]\n\n',
        ]),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )
    }) as typeof fetch

    const chunks: unknown[] = []
    for await (const chunk of queryModelLocalStreaming(baseQueryArgs)) {
      chunks.push(chunk)
    }
    const msg = chunks[0] as {
      message: { content: Array<Record<string, unknown>> }
    }
    const blocks = msg.message.content
    expect(blocks.find(b => b.type === 'tool_use')).toBeUndefined()
    const text = blocks.find(b => b.type === 'text') as
      | { text: string }
      | undefined
    expect(text?.text ?? '').toContain('invalid JSON arguments')
    expect(text?.text ?? '').toContain('bash')
  })

  // ---------------------------------------------------------------------
  // Fix 3: empty-turn nudge
  // ---------------------------------------------------------------------
  it('emits a retry-suggested nudge when the model returns tokens but no content', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        sseStream([
          sseEvent({
            choices: [{ delta: {} }],
            usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
          }),
          'data: [DONE]\n\n',
        ]),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )
    }) as typeof fetch

    const chunks: unknown[] = []
    for await (const chunk of queryModelLocalStreaming(baseQueryArgs)) {
      chunks.push(chunk)
    }
    const msg = chunks[0] as {
      message: { content: Array<Record<string, unknown>> }
    }
    const text = msg.message.content.find(b => b.type === 'text') as
      | { text: string }
      | undefined
    expect(text?.text ?? '').toContain('empty response')
  })
})

// -----------------------------------------------------------------------
// Direct unit tests for the exported pure helpers
// -----------------------------------------------------------------------
describe('recoverLeakedGemmaToolCalls', () => {
  it('parses a string-arg tool call', () => {
    const { cleanedText, recovered } = recoverLeakedGemmaToolCalls(
      '<|tool_call>call:Bash{command:<|"|>echo hi<|"|>}<tool_call|>',
    )
    expect(cleanedText).toBe('')
    expect(recovered).toEqual([{ name: 'Bash', args: { command: 'echo hi' } }])
  })

  it('parses multiple values and bare numbers/booleans', () => {
    const { recovered } = recoverLeakedGemmaToolCalls(
      '<|tool_call>call:fn{n:42,flag:true,name:<|"|>x<|"|>}<tool_call|>',
    )
    expect(recovered[0]?.args).toEqual({ n: 42, flag: true, name: 'x' })
  })

  it('passes through text without tool-call markers', () => {
    const { cleanedText, recovered } = recoverLeakedGemmaToolCalls(
      'just regular content',
    )
    expect(cleanedText).toBe('just regular content')
    expect(recovered).toHaveLength(0)
  })

  it('handles nested object args (validates non-greedy regex works for nesting)', () => {
    const { recovered } = recoverLeakedGemmaToolCalls(
      '<|tool_call>call:fn{outer:{inner_key:<|"|>v<|"|>,n:1}}<tool_call|>',
    )
    expect(recovered).toHaveLength(1)
    expect(recovered[0]?.args).toEqual({
      outer: { inner_key: 'v', n: 1 },
    })
  })
})

describe('repairToolArgsJson', () => {
  it('returns parsed object for valid JSON', () => {
    expect(repairToolArgsJson('{"a":1}')).toEqual({ a: 1 })
  })

  it('balances a missing closing brace', () => {
    expect(repairToolArgsJson('{"command":"ls"')).toEqual({ command: 'ls' })
  })

  it('strips a leading prose preamble', () => {
    expect(repairToolArgsJson('Here are the args: {"x":2}')).toEqual({ x: 2 })
  })

  it('removes trailing commas', () => {
    expect(repairToolArgsJson('{"a":1,"b":2,}')).toEqual({ a: 1, b: 2 })
  })

  it('returns null for unrecoverable garbage', () => {
    expect(repairToolArgsJson('!@#$%^&*')).toBeNull()
  })

  it('returns empty object for empty input', () => {
    expect(repairToolArgsJson('')).toEqual({})
  })
})

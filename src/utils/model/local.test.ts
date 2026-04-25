import { describe, expect, it } from 'bun:test'

import { getContextWindowForModel, getModelMaxOutputTokens } from '../context.js'
import {
  LOCAL_CONTEXT_WINDOW,
  LOCAL_DEFAULT_MODEL,
  LOCAL_MODEL_IDS,
  getLocalModelDisplayName,
  getLocalModelId,
  isLocalModel,
} from './local.js'

describe('local model defaults', () => {
  it('uses the promoted Qwen CTF alias by default', () => {
    expect(LOCAL_DEFAULT_MODEL).toBe('qwen3.6-27b-ctf')
  })

  it('keeps the Qwen3.6 native context window', () => {
    expect(LOCAL_CONTEXT_WINDOW).toBe(262144)
  })

  it('treats every bundled local router model as local', () => {
    for (const model of LOCAL_MODEL_IDS) {
      expect(isLocalModel(model)).toBe(true)
      expect(getLocalModelId(model)).toBe(model)
      expect(getContextWindowForModel(model)).toBe(LOCAL_CONTEXT_WINDOW)
      expect(getModelMaxOutputTokens(model)).toEqual({
        default: 32768,
        upperLimit: 32768,
      })
    }
  })

  it('keeps distinct display names for local router models', () => {
    expect(getLocalModelDisplayName('qwen3.6-27b-ctf')).toBe('Qwen 3.6 CTF')
    expect(getLocalModelDisplayName('qwen3.6-27b-dense')).toBe(
      'Qwen 3.6 27B Dense',
    )
    expect(getLocalModelDisplayName('gemma4-31b')).toBe('Gemma 4 31B')
  })
})

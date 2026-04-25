export const LOCAL_DEFAULT_MODEL =
  process.env.LOCAL_MODEL ?? 'qwen3.6-30b-a3b'
// Extra IDs recognized by isLocalModel() even if not the default.
// `qwen3.6-27b-ctf` is the planned CTF-promoted variant — kept here so
// selecting it later (once served) works without another code change.
export const LOCAL_MODEL_IDS = [
  LOCAL_DEFAULT_MODEL,
  'qwen3.6-27b-ctf',
  'qwen3.6-27b-dense',
  'gemma4-31b',
] as const
// Default: the LiteLLM router on port 4000, which fans out to one vLLM
// backend per model (see /project/inniang/inference/router/router.sh and
// /project/inniang/inference/router/litellm.config.yaml). The `model` field
// in each request selects the backend. To bypass the router and talk to a
// single vLLM backend directly (e.g. port 8000), set LOCAL_BASE_URL=...:8000/v1.
export const LOCAL_BASE_URL =
  process.env.LOCAL_BASE_URL ?? 'http://127.0.0.1:4000/v1'
// Promoted Qwen3.6-27B CTF deployment hard limit. Drives auto-compact
// threshold so compaction fires well before the 400 overflow rather than at
// the boundary.
export const LOCAL_CONTEXT_WINDOW =
  parseInt(process.env.LOCAL_CONTEXT_WINDOW ?? '262144', 10)

export function getLocalApiKey(): string {
  return process.env.LOCAL_API_KEY ?? 'local'
}

export function getLocalModelId(model: string): string | undefined {
  const normalized = model.toLowerCase()
  return LOCAL_MODEL_IDS.find(localModel =>
    normalized.includes(localModel.toLowerCase()),
  )
}

export function isLocalModel(model: string): boolean {
  return getLocalModelId(model) !== undefined
}

export function getLocalModelDisplayName(model: string): string | undefined {
  switch (getLocalModelId(model)) {
    case 'qwen3.6-30b-a3b':
      return 'Qwen 3.6 30B-A3B'
    case 'qwen3.6-27b-ctf':
      return 'Qwen 3.6 27B CTF'
    case 'qwen3.6-27b-dense':
      return 'Qwen 3.6 27B Dense'
    case 'gemma4-31b':
      return 'Gemma 4 31B'
    case LOCAL_DEFAULT_MODEL:
      return LOCAL_DEFAULT_MODEL
    default:
      return undefined
  }
}

export function isLocalProviderEnabled(): boolean {
  return true
}

/**
 * Local models that do not support the `tool_choice` parameter.
 * Sending `tool_choice: "auto"` to these models returns HTTP 400.
 */
export const LOCAL_MODELS_NO_TOOL_CHOICE = ['gemma4-31b'] as const

export function modelSupportsToolChoice(model: string): boolean {
  return !LOCAL_MODELS_NO_TOOL_CHOICE.some(
    id => model.toLowerCase().includes(id.toLowerCase()),
  )
}

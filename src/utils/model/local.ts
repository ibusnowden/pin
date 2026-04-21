export const LOCAL_DEFAULT_MODEL =
  process.env.LOCAL_MODEL ?? 'qwen3.6-30b-a3b'
export const LOCAL_BASE_URL =
  process.env.LOCAL_BASE_URL ?? 'http://127.0.0.1:8000/v1'

export function getLocalApiKey(): string {
  return process.env.LOCAL_API_KEY ?? 'local'
}

export function isLocalProviderEnabled(): boolean {
  return true
}

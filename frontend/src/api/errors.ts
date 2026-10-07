export interface ApiErrorInit {
  message: string
  status?: number
  detail?: unknown
  cause?: unknown
}

/** Normalized error shape every API helper rejects with. */
export class ApiError extends Error {
  readonly status?: number
  readonly detail?: unknown
  override readonly cause?: unknown

  constructor(init: ApiErrorInit) {
    super(init.message)
    this.name = 'ApiError'
    this.status = init.status
    this.detail = init.detail
    this.cause = init.cause
  }
}

/**
 * Turn any thrown/response value into a user-facing Chinese message.
 * Network failures get a stable message, HTTP failures prefer a backend
 * `error`/`detail` payload, and everything else falls back to a generic one.
 */
export function normalizeApiError(
  input: unknown,
  status?: number,
  fallback = '请求失败，请稍后重试',
): ApiError {
  if (input instanceof ApiError) return input

  if (typeof input === 'string') {
    const message = input.trim() || fallback
    return new ApiError({ message, status, detail: input })
  }

  if (input instanceof Error) {
    const isNetworkFailure =
      input.name === 'TypeError' || /failed to fetch|networkerror|load failed/i.test(input.message)
    return new ApiError({
      message: isNetworkFailure ? '网络连接失败，请确认后端服务已启动' : input.message || fallback,
      status,
      cause: input,
    })
  }

  if (input && typeof input === 'object') {
    const record = input as Record<string, unknown>
    const rawMessage = record.error ?? record.detail ?? record.message
    if (typeof rawMessage === 'string' && rawMessage.trim()) {
      return new ApiError({ message: rawMessage.trim(), status, detail: input })
    }
  }

  return new ApiError({ message: fallback, status, detail: input })
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

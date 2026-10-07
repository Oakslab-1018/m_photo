import {
  type AnalyzePointResponse,
  type ChatMessage,
  type CountriesResponse,
  type Country,
  type FeedbackPayload,
  type FeedbackResponse,
  type MessagesResponse,
  type Point,
  type UploadResponse,
} from '../types'
import { ApiError, normalizeApiError } from './errors'

export interface ApiClientOptions {
  baseUrl?: string
  fetchImpl?: typeof fetch
}

type JsonRecord = Record<string, unknown>

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : []
}

function asMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value) || typeof value.content !== 'string') return null
  return {
    id: typeof value.id === 'number' ? value.id : undefined,
    role: typeof value.role === 'string' ? value.role : 'ai',
    content: value.content,
    created_at: typeof value.created_at === 'string' || value.created_at === null ? value.created_at : undefined,
    meta: isRecord(value.meta) ? value.meta : null,
  }
}

export class GeoApi {
  private readonly baseUrl: string
  private readonly fetchImpl: typeof fetch

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? '').replace(/\/$/, '')
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis)
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response
    try {
      response = await this.fetchImpl(this.url(path), init)
    } catch (error) {
      throw normalizeApiError(error)
    }

    const contentType = response.headers.get('content-type') ?? ''
    let body: unknown = null
    try {
      body = contentType.includes('application/json') ? await response.json() : await response.text()
    } catch {
      body = null
    }

    if (!response.ok) {
      throw normalizeApiError(body, response.status, `请求失败（HTTP ${response.status}）`)
    }

    if (!isRecord(body)) {
      throw new ApiError({ message: '后端返回了无法识别的数据格式', status: response.status, detail: body })
    }

    if (typeof body.error === 'string' && body.error.trim()) {
      throw normalizeApiError(body, response.status)
    }

    return body as T
  }

  async health(): Promise<{ status?: string; healthy?: boolean }> {
    const body = await this.request<JsonRecord>('/api/health')
    return {
      status: typeof body.status === 'string' ? body.status : undefined,
      healthy: typeof body.healthy === 'boolean' ? body.healthy : responseOk(body),
    }
  }

  async upload(image: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('image', image)
    const raw = await this.request<JsonRecord>('/api/geo/upload', {
      method: 'POST',
      body: formData,
    })
    return {
      session_id: numberField(raw, 'session_id'),
      image_url: stringField(raw, 'image_url'),
      clues: asArray(raw.clues),
      scores: asArray(raw.scores),
      action: isRecord(raw.action) ? (raw.action as unknown as UploadResponse['action']) : null,
      messages: asArray(raw.messages).map(asMessage).filter((item): item is ChatMessage => item !== null),
    }
  }

  async countries(): Promise<Country[]> {
    const body = await this.request<CountriesResponse | JsonRecord>('/api/geo/countries')
    const record = body as JsonRecord
    return asArray<Country>(record.countries)
  }

  async feedback(sessionId: number, payload: FeedbackPayload): Promise<FeedbackResponse> {
    return this.request<FeedbackResponse>(`/api/geo/${sessionId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  }

  async messages(sessionId: number): Promise<ChatMessage[]> {
    const body = await this.request<MessagesResponse | JsonRecord>(`/api/geo/${sessionId}/messages`)
    const record = body as JsonRecord
    return asArray(record.messages).map(asMessage).filter((item): item is ChatMessage => item !== null)
  }

  async analyzePoint(sessionId: number, point: Point): Promise<AnalyzePointResponse> {
    return this.request<AnalyzePointResponse>(`/api/geo/${sessionId}/analyze-point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: point.x, y: point.y }),
    })
  }
}

function responseOk(body: JsonRecord): boolean {
  return body.ok === true || body.status === 'ok' || body.status === 'healthy'
}

function stringField(record: JsonRecord, key: string): string {
  const value = record[key]
  if (typeof value !== 'string') {
    throw new ApiError({ message: `后端响应缺少字段：${key}` })
  }
  return value
}

function numberField(record: JsonRecord, key: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ApiError({ message: `后端响应缺少字段：${key}` })
  }
  return value
}

export const api = new GeoApi()

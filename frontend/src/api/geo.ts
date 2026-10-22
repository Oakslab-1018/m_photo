import {
  type AnalyzePointResponse,
  type AnalyzePointRequest,
  type ChatMessage,
  type CitiesResponse,
  type ConfidenceRoute,
  type CountriesResponse,
  type Country,
  type FeedbackPayload,
  type FeedbackResponse,
  type GuidedDialogueResponse,
  type GuidedEventPayload,
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

function asConfidenceRoute(value: unknown): ConfidenceRoute | null {
  if (!isRecord(value) || !Array.isArray(value.top3)) return null
  if (value.tier !== 'high' && value.tier !== 'medium' && value.tier !== 'low') return null
  if (
    value.action !== 'direct_top1' &&
    value.action !== 'confirm_candidates' &&
    value.action !== 'ask_clue'
  ) {
    return null
  }
  return value as unknown as ConfidenceRoute
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
      throw new ApiError({ message: '返回的数据无法识别，请稍后重试', status: response.status, detail: body })
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
      confidence: asConfidenceRoute(raw.confidence),
      messages: asArray(raw.messages).map(asMessage).filter((item): item is ChatMessage => item !== null),
      inference_notice:
        typeof raw.inference_notice === 'string' && raw.inference_notice.trim()
          ? raw.inference_notice.trim()
          : null,
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

  /**
   * Fetch representative city candidates for one country. City points are
   * ranking diagnostics, never photo GPS; an unavailable backend (404/older
   * deployment) surfaces as a normalized ApiError for the caller to fall back on.
   *
   * The optional `sourceIso` filters one aggregated CN region scope
   * (CN/TW/HK/MO). Omitting it keeps the legacy merged-country request, so the
   * third argument stays backward compatible with older deployments.
   */
  async cities(sessionId: number, iso: string, sourceIso?: string | null): Promise<CitiesResponse> {
    const source = typeof sourceIso === 'string' ? sourceIso.trim().toUpperCase() : ''
    const payload: JsonRecord = source ? { iso, source_iso: source } : { iso }
    const raw = await this.request<JsonRecord>(`/api/geo/${sessionId}/cities`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return {
      city_scores: asArray<CitiesResponse['city_scores'][number]>(raw.city_scores),
      country_iso: typeof raw.country_iso === 'string' ? raw.country_iso : iso,
      scope: isRecord(raw.scope) ? (raw.scope as unknown as CitiesResponse['scope']) : null,
      status: typeof raw.status === 'string' ? (raw.status as CitiesResponse['status']) : 'ok',
      clue_dimensions: Array.isArray(raw.clue_dimensions)
        ? (raw.clue_dimensions.filter((item) => typeof item === 'string') as string[])
        : null,
      clue_note: typeof raw.clue_note === 'string' && raw.clue_note.trim() ? raw.clue_note.trim() : null,
      source_iso: typeof raw.source_iso === 'string' ? raw.source_iso : null,
      source_region: typeof raw.source_region === 'string' || isRecord(raw.source_region)
        ? (raw.source_region as CitiesResponse['source_region']) : null,
      source_isos: asArray<string>(raw.source_isos),
      note: typeof raw.note === 'string' && raw.note.trim() ? raw.note.trim() : null,
    }
  }

  async dialogueStart(sessionId: number): Promise<GuidedDialogueResponse> {
    return this.request<GuidedDialogueResponse>(`/api/geo/${sessionId}/dialogue/start`, {
      method: 'POST',
    })
  }

  async dialogue(sessionId: number): Promise<GuidedDialogueResponse> {
    return this.request<GuidedDialogueResponse>(`/api/geo/${sessionId}/dialogue`)
  }

  async dialogueEvent(
    sessionId: number,
    payload: GuidedEventPayload,
  ): Promise<GuidedDialogueResponse> {
    return this.request<GuidedDialogueResponse>(`/api/geo/${sessionId}/dialogue/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  }

  async analyzePoint(sessionId: number, request: AnalyzePointRequest | Point): Promise<AnalyzePointResponse> {
    const payload: AnalyzePointRequest = { x: request.x, y: request.y }
    if ('scale' in request && request.scale !== undefined) payload.scale = request.scale
    if ('image_region' in request && request.image_region !== undefined) payload.image_region = request.image_region
    if ('purpose' in request && request.purpose !== undefined) payload.purpose = request.purpose
    if ('client_request_id' in request && request.client_request_id !== undefined) {
      payload.client_request_id = request.client_request_id
    }
    return this.request<AnalyzePointResponse>(`/api/geo/${sessionId}/analyze-point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  }
}

function responseOk(body: JsonRecord): boolean {
  return body.ok === true || body.status === 'ok' || body.status === 'healthy'
}

function stringField(record: JsonRecord, key: string): string {
  const value = record[key]
  if (typeof value !== 'string') {
    throw new ApiError({ message: `返回的数据缺少「${key}」` })
  }
  return value
}

function numberField(record: JsonRecord, key: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ApiError({ message: `返回的数据缺少「${key}」` })
  }
  return value
}

export const api = new GeoApi()

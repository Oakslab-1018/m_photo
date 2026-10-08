export type ClueDimension =
  | 'language'
  | 'signage'
  | 'vegetation'
  | 'building'
  | 'vehicle'
  | 'climate'
  | 'culture'
  | 'other'

export interface Clue {
  dimension: ClueDimension | string
  value: string
  confidence: number
  evidence?: string
  source?: string
  extra?: Record<string, unknown>
}

export type MessageRole = 'ai' | 'user' | 'system' | string

export interface ChatMessage {
  id?: number
  role: MessageRole
  content: string
  created_at?: string | null
  meta?: Record<string, unknown> | null
}

export interface CountryScore {
  iso: string
  name: string
  name_zh: string
  score: number
  raw_cosine?: number | null
  score_type?: string
  matched_clues?: Array<Record<string, unknown>>
  lat: number
  lng: number
}

export interface Country extends CountryScore {
  continent?: string
  driving_side?: string
  languages_spoken?: string[]
  features?: Record<string, string[]>
}

export type ActionKind = 'confirm' | 'ask' | 'request_clue' | 'done' | string

export interface HypothesisAction {
  kind: ActionKind
  message?: string
  payload?: {
    iso?: string
    name_zh?: string
    score?: number
    dimensions?: string[]
    candidates?: string[]
    [key: string]: unknown
  }
}

export interface UploadResponse {
  session_id: number
  image_url: string
  clues: Clue[]
  scores: CountryScore[]
  action: HypothesisAction | null
  messages: ChatMessage[]
}

export interface FeedbackResponse {
  clues?: Clue[]
  scores?: CountryScore[]
  action?: HypothesisAction | null
  messages?: ChatMessage[]
  exclusions?: string[]
  done?: boolean
  iso?: string
}

export interface AnalyzePointResponse {
  clues?: Clue[]
  scores?: CountryScore[]
  action?: HypothesisAction | null
  messages?: ChatMessage[]
  exclusions?: string[]
  new_clues?: Clue[]
  crop_info?: { x: number; y: number; size: number }
}

export interface MessagesResponse {
  messages: ChatMessage[]
}

export interface CountriesResponse {
  countries: Country[]
}

export type FeedbackType = 'confirm' | 'reject' | 'answer' | 'free_text'

export interface FeedbackPayload {
  type: FeedbackType
  iso?: string
  dimension?: string
  value?: string
  text?: string
}

export interface Point {
  x: number
  y: number
}

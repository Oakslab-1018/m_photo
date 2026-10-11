export type ClueDimension =
  | 'language'
  | 'signage'
  | 'place_name'
  | 'road'
  | 'driving_side'
  | 'terrain'
  | 'waterbody'
  | 'vegetation'
  | 'building'
  | 'vehicle'
  | 'climate'
  | 'culture'
  | 'infrastructure'
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

export interface ReferenceMatch {
  reference_id: string
  iso: string
  region: string
  place: string
  viewpoint?: string
  city?: string
  city_id?: string
  lat?: number
  lng?: number
  raw_cosine: number
}

export interface ReferenceRegion {
  iso: string
  region: string
  raw_cosine: number
  prototype_count: number
  support: number
}

export type CityScoreType = 'city_text_cosine' | 'city_reference_hybrid' | 'city_clue_fusion' | string

export interface CityScore {
  city_id: string
  iso: string
  name: string
  region: string
  lat: number
  lng: number
  raw_cosine: number
  score: number
  score_type: CityScoreType
  reference_cosine?: number | null
  /** Optional clue-fusion diagnostics for city_clue_fusion rows; shape owned by the backend contract. */
  matched_clues?: Array<Record<string, unknown>>
  fusion?: Record<string, unknown>
}

/** Scope counters; every field optional because an unavailable response may send {} */
export interface CityScope {
  city_limit_per_country?: number
  city_catalog_revision?: string
  cities_considered?: number
}

export type CitiesStatus = 'ok' | 'unavailable' | 'no_cities'

export interface CitiesResponse {
  city_scores: CityScore[]
  country_iso: string
  scope?: CityScope | null
  status: CitiesStatus
  clue_dimensions?: string[] | null
  clue_note?: string | null
}

export interface CountryScore {
  iso: string
  name: string
  name_zh: string
  score: number
  raw_cosine?: number | null
  score_type?: string
  /** Optional opaque fusion diagnostics; shape owned by the backend contract. */
  fusion?: Record<string, unknown>
  /** Metadata-only labeled-reference diagnostics; never photo GPS. */
  reference_matches?: ReferenceMatch[]
  reference_regions?: ReferenceRegion[]
  reference_cosine?: number | null
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

/** Structured answer choice attached to a pending dialogue action. */
export interface ClueAnswerOption {
  dimension: string
  value: string
  label?: string
}

export interface HypothesisAction {
  kind: ActionKind
  message?: string
  payload?: {
    iso?: string
    name_zh?: string
    score?: number
    dimensions?: string[]
    candidates?: string[]
    /** Answer choices for the pending question; the AI may question or reason. */
    options?: ClueAnswerOption[]
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
  city_scores?: CityScore[]
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

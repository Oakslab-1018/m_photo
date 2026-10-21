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

/**
 * One raw regional source aggregated under country CN (TW/HK/MO), optionally
 * with member diagnostics. Unknown extra keys are preserved untouched.
 */
export interface SourceRegionRef {
  source_iso?: string | null
  iso?: string | null
  name_zh?: string | null
  parent_iso?: string | null
  [key: string]: unknown
}

export type SourceRegionEntry = string | SourceRegionRef
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
  /** CN aggregate metadata: TW/HK/MO city rows keep their raw source region. */
  country_iso?: string | null
  source_iso?: string | null
  source_region?: SourceRegionEntry | null
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
  /** Present when the request filtered one CN region scope (e.g. source_iso TW). */
  source_iso?: string | null
  source_region?: SourceRegionEntry | null
  source_isos?: string[]
  note?: string | null
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
  /**
   * Geopolicy aggregation metadata. The backend may expose TW as an upstream
   * region under the canonical CN country row; the shared geo utility folds any
   * leftover raw TW row into CN before display.
   */
  parent_iso?: string | null
  is_region?: boolean
  is_grouped?: boolean
  source_regions?: SourceRegionEntry[]
  /** Optional backend-provided region-scope label (e.g. 中国台湾). */
  region_scope?: string | null
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
    /** Server-owned country routing metadata mirrored by guided responses. */
    confidence?: ConfidenceRoute | null
    /**
     * Optional server receipt for the last submitted feedback; status is
     * `applied | unchanged | unrecognized` plus a human summary.
     */
    feedback_receipt?: Record<string, unknown> | null
    [key: string]: unknown
  }
}

export type ConfidenceTier = 'high' | 'medium' | 'low'
export type ConfidenceAction = 'direct_top1' | 'confirm_candidates' | 'ask_clue'

/** Confidence top3 rows are intentionally sparse ranking summaries, not full country rows. */
export type ConfidenceCountryScore = Partial<CountryScore> & Pick<CountryScore, 'iso'>

/** Deterministic routing metadata; margins are ranking gaps, not probabilities. */
export interface ConfidenceRoute {
  tier: ConfidenceTier
  action: ConfidenceAction
  model_only_margin?: number | null
  fused_margin?: number | null
  effective_margin?: number | null
  prior_top_iso?: string | null
  top1_iso?: string | null
  top3: ConfidenceCountryScore[]
  leader_changed: boolean
  reason: string
  policy_version: string
}

export interface UploadResponse {
  session_id: number
  image_url: string
  clues: Clue[]
  scores: CountryScore[]
  action: HypothesisAction | null
  confidence?: ConfidenceRoute | null
  messages: ChatMessage[]
  city_scores?: CityScore[]
  /** Natural Chinese explanation when inference degraded; null/absent when healthy. */
  inference_notice?: string | null
}

export interface FeedbackResponse {
  clues?: Clue[]
  scores?: CountryScore[]
  action?: HypothesisAction | null
  confidence?: ConfidenceRoute | null
  messages?: ChatMessage[]
  exclusions?: string[]
  done?: boolean
  iso?: string
}

export interface AnalyzePointResponse {
  clues?: Clue[]
  scores?: CountryScore[]
  action?: HypothesisAction | null
  confidence?: ConfidenceRoute | null
  messages?: ChatMessage[]
  exclusions?: string[]
  new_clues?: Clue[]
  crop_info?: CropInfo
  focus_region?: RegionReport | null
  model_status?: RegionModelStatus | null
  ocr_status?: RegionOcrStatus | null
  applied?: boolean
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

/**
 * Client-side optimistic chat entry. Rendering and retry live entirely in the
 * UI layer; it is replaced exactly once by the authoritative server messages.
 */
export interface PendingChatMessage {
  id: number
  role: 'user'
  content: string
  status: 'sending' | 'failed'
  error?: string | null
  payload: FeedbackPayload
  /** Session the message was submitted against; null when no session existed. */
  sessionId?: number | null
  /** UI epoch at submission time; superseded requests stay visible as failures. */
  epoch?: number
  /**
   * True when the session/epoch this bubble belonged to was replaced. Retry is
   * blocked so old-photo feedback is never silently sent to a new session.
   */
  orphaned?: boolean
}

export interface Point {
  x: number
  y: number
}

export type AnalyzePurpose = 'auto' | 'ocr' | 'scene'

export interface ImageRegion {
  x: number
  y: number
  w: number
  h: number
}

export interface AnalyzePointRequest extends Point {
  scale?: number
  image_region?: ImageRegion | null
  purpose?: AnalyzePurpose
  client_request_id?: string | null
}

export interface RegionBBox {
  x: number
  y: number
  w: number
  h: number
}

export type RegionObservationKind =
  | 'ocr_text'
  | 'semantic'
  | 'pixel_signal'
  | 'scene_hypothesis'
  | 'crop_ranking'

export type RegionTrustLevel = 'high' | 'medium' | 'low'

export interface RegionTrust {
  level: RegionTrustLevel
  reliable: boolean
  validated: boolean
  affects_ranking: boolean
  confidence_kind?: string | null
  confidence?: number | null
  raw_cosine?: number | null
  uncertain?: boolean | null
}

export interface RegionObservation {
  id: string
  kind: RegionObservationKind
  dimension: string
  value: string
  text?: string | null
  label: string
  evidence: string
  source: string
  provider?: string | null
  bbox_norm?: RegionBBox | null
  quad_norm?: RegionBBox | null
  trust: RegionTrust
}

export interface RegionQuestion {
  id: string
  text: string
  reason: string
  basis_observation_ids: string[]
}

export interface RegionQuality {
  issues: string[]
  metrics: Record<string, unknown>
}

export interface RegionImage {
  width: number
  height: number
  orientation_applied: boolean
  detail_width: number
  detail_height: number
  ocr_scale: number
}

export interface RegionOcrStatus {
  provider: string
  enabled: boolean
  available: boolean
  state: string
  line_count: number
}

export interface RegionModelStatus {
  provider: string
  enabled: boolean
  available: boolean
  state: string
  ranking_applied: boolean
  margin?: number | null
  uncertain?: boolean | null
  scene_available?: boolean | null
  scene_state?: string | null
  ranking_note?: string | null
}

export type RegionDedupe = 'new' | 'duplicate_region' | 'duplicate_request' | 'no_new_evidence'

export interface RegionReport {
  id: string
  purpose: AnalyzePurpose
  scale: number
  bbox_px: number[]
  bbox_norm: RegionBBox
  context_bbox_norm?: RegionBBox | null
  image: RegionImage
  quality: RegionQuality
  ocr: RegionOcrStatus
  model: RegionModelStatus
  observations: RegionObservation[]
  question_suggestions: RegionQuestion[]
  summary: string
  dedupe: RegionDedupe
}

export interface CropInfo extends Point {
  size: number
  region_id?: string | null
  bbox_px?: number[] | null
  bbox_norm?: RegionBBox | null
  scale?: number | null
  purpose?: AnalyzePurpose | null
}

export interface RegionAnalysisRecord {
  id: string
  request: AnalyzePointRequest
  status: 'pending' | 'failed'
  error?: string | null
  sessionId: number
  epoch: number
}
/* ------------------------------------------------------------------ */
/* Guided dialogue contract (see docs/guided-dialogue.md)             */
/* ------------------------------------------------------------------ */

/** Public guided path is country -> city -> resolved. */
export type GuidedStage = 'country' | 'city' | 'resolved'
/** Retained for deprecated persisted admin1 metadata; never a public UI step. */
export type GuidedLocationLevel = Exclude<GuidedStage, 'resolved'> | 'admin1'

/**
 * UI phase inside one guided stage.
 * - review: one server-pinned observation/question at a time, candidates hidden.
 * - candidates: user explicitly accepted the review round and may act on Top-5.
 */
export type GuidedPhase = 'review' | 'candidates'

/** Minimal target reference used by guided events. */
export interface GuidedTargetRef {
  level: GuidedLocationLevel
  id: string
}

/** Catalog-backed location target. IDs are authoritative and never inferred by the UI. */
export interface LocationTarget extends GuidedTargetRef {
  label: string
  country_iso: string
  source_iso?: string | null
  admin1_id?: string | null
  city_id?: string | null
  lat?: number | null
  lng?: number | null
  coordinate_source?: string | null
  coordinate_quality?: string | null
  coordinate_city_id?: number | null
}

export interface PlaceSource {
  provider: string
  provider_place_id?: string | null
  label: string
  query?: string
  attribution?: string | null
  license?: string | null
  fetched_at?: string
}

export interface PlaceMatch {
  query_terms: string[]
  clue_ids: string[]
  match_kind: 'catalog_exact' | 'geocoder_locality'
  ambiguous: boolean
  sources: PlaceSource[]
}

/** One ranked/sorted location option. score is absent for catalog-only rows. */
export interface LocationCandidate {
  target: LocationTarget
  score?: number | null
  score_type?: string | null
  reason?: string | null
  /** Server-authorized provenance; explicit matches outrank model score rows. */
  selection_basis?: 'user_text' | 'catalog_search'
  place_match?: PlaceMatch | null
}

export type ReviewedClueStatus = 'pending' | 'confirmed' | 'rejected' | 'uncertain'

/** Evidence review state is separate from geographic exclusion state. */
export interface ReviewedClue {
  id: string
  dimension: string
  value: string
  description: string
  source: string
  confidence?: number | null
  status: ReviewedClueStatus
}

export interface GuidedQuestionOption {
  value: string
  label: string
}

export type GuidedQuestionKind = 'review_clue' | 'ask' | 'confirm_location' | 'request_clue'

export interface GuidedQuestion {
  id: string
  kind: GuidedQuestionKind
  message: string
  dimension?: string | null
  clue_id?: string | null
  /** Present for confirm_location; optional for older/adjacent backend payloads. */
  target?: LocationTarget | null
  options?: GuidedQuestionOption[] | null
}

export interface GuidedConfirmedPath {
  country: LocationTarget | null
  /** Deprecated legacy metadata; preserved but ignored by path/markers. */
  admin1?: LocationTarget | null
  city: LocationTarget | null
}

export interface GuidedDialogue {
  version: 1
  revision: number
  stage: GuidedStage
  confirmed: GuidedConfirmedPath
  exclusions: LocationTarget[]
  clues: ReviewedClue[]
  pending_question: GuidedQuestion | null
}

export type GuidedEventType =
  | 'confirm_clue'
  | 'reject_clue'
  | 'confirm_location'
  | 'reject_location'
  | 'answer'
  | 'unknown'
  | 'retract'
  | 'backtrack'
  | 'free_text'

/** Event sent to the authoritative dialogue endpoint. */
export interface GuidedEventPayload {
  type: GuidedEventType
  event_id: string
  expected_revision: number
  question_id?: string
  target?: GuidedTargetRef
  clue_id?: string
  value?: string
  text?: string
  level?: GuidedLocationLevel
}

export interface GuidedEventReceipt {
  event_id: string
  status: string
  summary?: string | null
}

/** Every guided endpoint returns this envelope. */
export interface GuidedDialogueResponse {
  dialogue: GuidedDialogue
  candidates: LocationCandidate[]
  action: HypothesisAction | null
  confidence?: ConfidenceRoute | null
  messages: ChatMessage[]
  scores: CountryScore[]
  clues: Clue[]
  exclusions: string[]
  receipt?: GuidedEventReceipt | null
  /** Natural Chinese explanation when inference degraded; null/absent when healthy. */
  inference_notice?: string | null
}

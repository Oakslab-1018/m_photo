import type {
  AnalyzePointRequest,
  AnalyzePurpose,
  ChatMessage,
  ImageRegion,
  RegionBBox,
  RegionDedupe,
  RegionImage,
  RegionModelStatus,
  RegionObservation,
  RegionObservationKind,
  RegionOcrStatus,
  RegionQuestion,
  RegionReport,
  RegionTrust,
  RegionTrustLevel,
} from '@/types'

const PURPOSES: AnalyzePurpose[] = ['auto', 'ocr', 'scene']
const OBSERVATION_KINDS: RegionObservationKind[] = [
  'ocr_text',
  'semantic',
  'pixel_signal',
  'scene_hypothesis',
  'crop_ranking',
]
const TRUST_LEVELS: RegionTrustLevel[] = ['high', 'medium', 'low']
const DEDUPE_VALUES: RegionDedupe[] = [
  'new',
  'duplicate_region',
  'duplicate_request',
  'no_new_evidence',
]

let requestIdCounter = 0

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function positiveNumber(value: unknown): number | null {
  const number = finiteNumber(value)
  return number !== null && number > 0 ? number : null
}

function nonNegativeInteger(value: unknown): number | null {
  const number = finiteNumber(value)
  return number !== null && Number.isInteger(number) && number >= 0 ? number : null
}

function cleanString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function boolValue(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item: unknown) => cleanString(item))
    .filter((item): item is string => item !== null)
}

export function normalizeImageRegion(value: unknown): ImageRegion | null {
  if (!isRecord(value)) return null
  const x = finiteNumber(value.x)
  const y = finiteNumber(value.y)
  const w = finiteNumber(value.w)
  const h = finiteNumber(value.h)
  if (x === null || y === null || w === null || h === null) return null
  if (x < 0 || y < 0 || w <= 0 || h <= 0) return null
  if (x + w > 1 + 1e-9 || y + h > 1 + 1e-9) return null
  return { x, y, w, h }
}

export function normalizeRegionBBox(value: unknown): RegionBBox | null {
  return normalizeImageRegion(value)
}

function normalizeTrust(value: unknown): RegionTrust | null {
  if (!isRecord(value)) return null
  const level = cleanString(value.level)
  if (!level || !TRUST_LEVELS.includes(level as RegionTrustLevel)) return null
  return {
    level: level as RegionTrustLevel,
    reliable: boolValue(value.reliable),
    validated: boolValue(value.validated),
    affects_ranking: boolValue(value.affects_ranking),
    confidence_kind: cleanString(value.confidence_kind),
    confidence: finiteNumber(value.confidence),
    raw_cosine: finiteNumber(value.raw_cosine),
    uncertain: typeof value.uncertain === 'boolean' ? value.uncertain : null,
  }
}

function normalizeObservation(value: unknown): RegionObservation | null {
  if (!isRecord(value)) return null
  const id = cleanString(value.id)
  const kind = cleanString(value.kind)
  const dimension = cleanString(value.dimension)
  const observationValue = cleanString(value.value)
  const label = cleanString(value.label)
  const evidence = cleanString(value.evidence)
  const source = cleanString(value.source)
  const trust = normalizeTrust(value.trust)
  if (
    !id ||
    !kind ||
    !OBSERVATION_KINDS.includes(kind as RegionObservationKind) ||
    !dimension ||
    !observationValue ||
    !label ||
    !evidence ||
    !source ||
    !trust
  ) {
    return null
  }
  return {
    id,
    kind: kind as RegionObservationKind,
    dimension,
    value: observationValue,
    text: cleanString(value.text),
    label,
    evidence,
    source,
    provider: cleanString(value.provider),
    bbox_norm: normalizeRegionBBox(value.bbox_norm),
    quad_norm: normalizeRegionBBox(value.quad_norm),
    trust,
  }
}

function normalizeQuestions(value: unknown): RegionQuestion[] {
  if (!Array.isArray(value)) return []
  const out: RegionQuestion[] = []
  for (const item of value) {
    if (!isRecord(item)) continue
    const id = cleanString(item.id)
    const text = cleanString(item.text)
    const reason = cleanString(item.reason)
    if (!id || !text || !reason) continue
    out.push({
      id,
      text,
      reason,
      basis_observation_ids: stringList(item.basis_observation_ids),
    })
  }
  return out
}

function normalizeImage(value: unknown): RegionImage | null {
  if (!isRecord(value)) return null
  const width = positiveNumber(value.width)
  const height = positiveNumber(value.height)
  const detailWidth = positiveNumber(value.detail_width)
  const detailHeight = positiveNumber(value.detail_height)
  const ocrScale = positiveNumber(value.ocr_scale)
  if (width === null || height === null || detailWidth === null || detailHeight === null || ocrScale === null) {
    return null
  }
  return {
    width: Math.round(width),
    height: Math.round(height),
    orientation_applied: boolValue(value.orientation_applied),
    detail_width: Math.round(detailWidth),
    detail_height: Math.round(detailHeight),
    ocr_scale: ocrScale,
  }
}

function normalizeOcrStatus(value: unknown): RegionOcrStatus {
  const raw: Record<string, unknown> = isRecord(value) ? value : {}
  return {
    provider: cleanString(raw.provider) ?? '未报告',
    enabled: boolValue(raw.enabled),
    available: boolValue(raw.available),
    state: cleanString(raw.state) ?? 'unknown',
    line_count: nonNegativeInteger(raw.line_count) ?? 0,
  }
}

function normalizeModelStatus(value: unknown): RegionModelStatus {
  const raw: Record<string, unknown> = isRecord(value) ? value : {}
  return {
    provider: cleanString(raw.provider) ?? '未报告',
    enabled: boolValue(raw.enabled),
    available: boolValue(raw.available),
    state: cleanString(raw.state) ?? 'unknown',
    ranking_applied: boolValue(raw.ranking_applied),
    margin: finiteNumber(raw.margin),
    uncertain: typeof raw.uncertain === 'boolean' ? raw.uncertain : null,
    scene_available: typeof raw.scene_available === 'boolean' ? raw.scene_available : null,
    scene_state: cleanString(raw.scene_state),
    ranking_note: cleanString(raw.ranking_note),
  }
}

export function normalizeRegionReport(value: unknown): RegionReport | null {
  if (!isRecord(value)) return null
  const id = cleanString(value.id)
  const bboxNorm = normalizeRegionBBox(value.bbox_norm)
  const image = normalizeImage(value.image)
  if (!id || !bboxNorm || !image) return null
  const purposeValue = cleanString(value.purpose)
  const purpose = purposeValue && PURPOSES.includes(purposeValue as AnalyzePurpose)
    ? (purposeValue as AnalyzePurpose)
    : 'auto'
  const dedupeValue = cleanString(value.dedupe)
  const dedupe = dedupeValue && DEDUPE_VALUES.includes(dedupeValue as RegionDedupe)
    ? (dedupeValue as RegionDedupe)
    : 'no_new_evidence'
  const qualityRaw: Record<string, unknown> = isRecord(value.quality) ? value.quality : {}
  const observations = Array.isArray(value.observations)
    ? value.observations
      .map((item: unknown) => normalizeObservation(item))
      .filter((item): item is RegionObservation => item !== null)
    : []
  return {
    id,
    purpose,
    scale: finiteNumber(value.scale) ?? 0.25,
    bbox_px: Array.isArray(value.bbox_px)
      ? value.bbox_px
        .map((item: unknown) => finiteNumber(item))
        .filter((item): item is number => item !== null)
        .map((item: number) => Math.round(item))
      : [],
    bbox_norm: bboxNorm,
    context_bbox_norm: normalizeRegionBBox(value.context_bbox_norm),
    image,
    quality: {
      issues: stringList(qualityRaw.issues),
      metrics: isRecord(qualityRaw.metrics) ? qualityRaw.metrics : {} as Record<string, unknown>,
    },
    ocr: normalizeOcrStatus(value.ocr),
    model: normalizeModelStatus(value.model),
    observations,
    question_suggestions: normalizeQuestions(value.question_suggestions),
    summary: cleanString(value.summary) ?? '',
    dedupe,
  }
}

/** Read persisted RegionReports from AI messages, deduped by authoritative report id. */
export function regionReportsFromMessages(messages: readonly ChatMessage[]): RegionReport[] {
  const order: string[] = []
  const byId = new Map<string, RegionReport>()
  for (const message of messages) {
    if (message.role !== 'ai' || !message.meta) continue
    const report = normalizeRegionReport(message.meta.point_analysis)
    if (!report) continue
    if (!byId.has(report.id)) order.push(report.id)
    byId.set(report.id, report)
  }
  return order.map((id) => byId.get(id)).filter((report): report is RegionReport => report !== undefined)
}

/** Merge reports while preserving first-seen history order and deduping ids. */
export function mergeRegionReports(...lists: ReadonlyArray<readonly RegionReport[]>): RegionReport[] {
  const order: string[] = []
  const byId = new Map<string, RegionReport>()
  for (const list of lists) {
    for (const report of list) {
      if (!byId.has(report.id)) order.push(report.id)
      byId.set(report.id, report)
    }
  }
  return order.map((id) => byId.get(id)).filter((report): report is RegionReport => report !== undefined)
}

/** UUID when available, otherwise the same compact timestamp/counter style as guided events. */
export function newRegionRequestId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID()
  requestIdCounter += 1
  return 'region-' + Date.now().toString(36) + '-' + requestIdCounter
}

export function observationBox(observation: RegionObservation): RegionBBox | null {
  return observation.bbox_norm ?? observation.quad_norm ?? null
}

export function normalizeAnalyzePointRequest(request: AnalyzePointRequest): AnalyzePointRequest | null {
  const x = finiteNumber(request.x)
  const y = finiteNumber(request.y)
  if (x === null || y === null || x < 0 || x > 1 || y < 0 || y > 1) return null
  const scaleValue = finiteNumber(request.scale)
  const scale = Math.min(0.8, Math.max(0.08, scaleValue ?? 0.25))
  const imageRegion = request.image_region ? normalizeImageRegion(request.image_region) : null
  const purpose = request.purpose && PURPOSES.includes(request.purpose) ? request.purpose : 'auto'
  const clientRequestId = cleanString(request.client_request_id)
  const normalized: AnalyzePointRequest = { x, y, scale, purpose }
  if (imageRegion) normalized.image_region = imageRegion
  if (clientRequestId) normalized.client_request_id = clientRequestId
  return normalized
}

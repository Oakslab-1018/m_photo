import type {
  PlaceMatch,
  PlaceObservationSource,
  PlaceSource,
  PlaceSourceType,
} from '@/types'

const SOURCE_TYPE_LABELS: Record<PlaceSourceType, string> = {
  local_catalog: '本地地名库',
  local_poi: '本地POI库',
  local_road: '本地道路库',
  web_search: '联网网页检索',
  geocoder: '联网地名检索',
}

const LOCAL_SOURCE_TYPES = new Set<PlaceSourceType>([
  'local_catalog',
  'local_poi',
  'local_road',
])

const WEB_SOURCE_TYPES = new Set<PlaceSourceType>(['web_search', 'geocoder'])

const OBSERVATION_SOURCE_LABELS: Record<PlaceObservationSource, string> = {
  ocr: 'OCR文字',
  user: '用户补充',
}

const CONTROL_CHARACTERS = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g

function unknownRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function nullableText(value: unknown): string | null {
  const text = cleanText(value)
  return text || null
}

function normalizeStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const result: string[] = []
  for (const item of value) {
    const text = cleanText(item)
    if (text && !result.includes(text)) result.push(text)
  }
  return result
}

function knownSourceType(value: unknown): PlaceSourceType | null {
  const type = cleanText(value).toLowerCase()
  if (type === 'local_catalog' || type === 'local_poi' || type === 'local_road') {
    return type
  }
  if (type === 'web_search' || type === 'geocoder') return type
  return null
}

function knownObservationSource(value: unknown): PlaceObservationSource | null {
  const source = cleanText(value).toLowerCase()
  if (source === 'ocr' || source === 'user') return source
  return null
}

/** Clean untrusted metadata and bound it before it reaches the template. */
export function boundedSourceText(value: unknown, limit = 280): string {
  const text = cleanText(value).replace(CONTROL_CHARACTERS, ' ').replace(/\s+/g, ' ')
  if (!text) return ''
  const safeLimit = Math.max(1, Math.floor(limit))
  return text.length > safeLimit
    ? `${text.slice(0, Math.max(1, safeLimit - 1)).trimEnd()}…`
    : text
}

/** Only absolute http(s) URLs without embedded credentials may become links. */
export function safeHttpSourceUrl(value: unknown): string | null {
  const text = cleanText(value)
  if (!text || text.length > 4096 || /[\u0000-\u0020\u007f]/.test(text)) return null
  if (!/^https?:\/\//i.test(text)) return null
  try {
    const parsed = new URL(text)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null
    if (parsed.username || parsed.password || !parsed.hostname) return null
    return parsed.href
  } catch {
    return null
  }
}

/** Preserve known and forward-compatible source fields without trusting them. */
export function normalizePlaceSource(value: unknown): PlaceSource | null {
  const raw = unknownRecord(value)
  if (!raw) return null
  const normalized: PlaceSource = {
    ...raw,
    provider: cleanText(raw.provider),
    label: cleanText(raw.label),
    provider_place_id: nullableText(raw.provider_place_id),
    attribution: nullableText(raw.attribution),
    license: nullableText(raw.license),
    source_type: nullableText(raw.source_type),
    url: nullableText(raw.url),
    title: nullableText(raw.title),
    excerpt: nullableText(raw.excerpt),
    observation_source: nullableText(raw.observation_source),
    fetched_at: nullableText(raw.fetched_at),
  }
  const query = cleanText(raw.query)
  if (query) normalized.query = query
  else delete normalized.query
  return normalized
}

export function placeSourceIdentity(source: PlaceSource): string {
  return JSON.stringify([
    source.provider,
    source.provider_place_id ?? '',
    source.label,
    source.query ?? '',
    source.source_type ?? '',
    source.url ?? '',
    source.title ?? '',
    source.excerpt ?? '',
    source.observation_source ?? '',
    source.fetched_at ?? '',
    source.attribution ?? '',
    source.license ?? '',
  ])
}

export function normalizePlaceSources(value: unknown): PlaceSource[] {
  if (!Array.isArray(value)) return []
  const sources: PlaceSource[] = []
  const seen = new Set<string>()
  for (const item of value) {
    const source = normalizePlaceSource(item)
    if (!source) continue
    const key = placeSourceIdentity(source)
    if (seen.has(key)) continue
    seen.add(key)
    sources.push(source)
  }
  return sources
}

export function mergePlaceSources(
  ...groups: Array<readonly PlaceSource[] | null | undefined>
): PlaceSource[] {
  return normalizePlaceSources(groups.flatMap((group) => group ?? []))
}

export function normalizePlaceMatch(value: unknown): PlaceMatch | null {
  const raw = unknownRecord(value)
  if (!raw) return null
  const matchKind = cleanText(raw.match_kind)
  const geocoderCandidate =
    matchKind === 'geocoder_candidate' || matchKind === 'geocoder_locality'
  const webCandidate = matchKind === 'web_candidate'
  const normalized: PlaceMatch = {
    ...raw,
    query_terms: normalizeStringList(raw.query_terms),
    clue_ids: normalizeStringList(raw.clue_ids),
    match_kind: geocoderCandidate
      ? 'geocoder_candidate'
      : matchKind
        ? (matchKind as PlaceMatch['match_kind'])
        : 'catalog_exact',
    ambiguous: raw.ambiguous === true,
    sources: normalizePlaceSources(raw.sources),
  }
  if (geocoderCandidate || webCandidate || raw.candidate_only === true) {
    normalized.candidate_only = true
  } else if (raw.candidate_only === false) {
    normalized.candidate_only = false
  } else {
    delete normalized.candidate_only
  }
  return normalized
}

export function mergePlaceMatches(
  ...values: Array<PlaceMatch | null | undefined>
): PlaceMatch | null {
  const matches = values
    .map(normalizePlaceMatch)
    .filter((match): match is PlaceMatch => match !== null)
  if (!matches.length) return null
  const merged: PlaceMatch = { ...matches[0] }
  for (const match of matches.slice(1)) {
    for (const [key, value] of Object.entries(match)) {
      if (
        key === 'query_terms' ||
        key === 'clue_ids' ||
        key === 'sources' ||
        key === 'ambiguous' ||
        key === 'candidate_only'
      ) {
        continue
      }
      if (merged[key] === undefined || merged[key] === null || merged[key] === '') {
        merged[key] = value
      }
    }
  }
  merged.query_terms = [...new Set(matches.flatMap((match) => match.query_terms))]
  merged.clue_ids = [...new Set(matches.flatMap((match) => match.clue_ids))]
  merged.sources = mergePlaceSources(...matches.map((match) => match.sources))
  merged.ambiguous = matches.some((match) => match.ambiguous)
  if (matches.some((match) => match.candidate_only === true)) merged.candidate_only = true
  else if (matches.some((match) => match.candidate_only === false)) merged.candidate_only = false
  else delete merged.candidate_only
  return merged
}

export function placeSourceTypeLabel(source: PlaceSource): string {
  const type = knownSourceType(source.source_type)
  if (type) return SOURCE_TYPE_LABELS[type]
  if (safeHttpSourceUrl(source.url)) return '联网来源'
  return '来源待核实'
}

export function isWebPlaceSource(source: PlaceSource): boolean {
  const type = knownSourceType(source.source_type)
  return type !== null && WEB_SOURCE_TYPES.has(type)
}

function placeSourceGroup(source: PlaceSource): 'local' | 'web' | 'unknown' {
  const type = knownSourceType(source.source_type)
  if (type && LOCAL_SOURCE_TYPES.has(type)) return 'local'
  if (type && WEB_SOURCE_TYPES.has(type)) return 'web'
  if (safeHttpSourceUrl(source.url)) return 'web'
  return 'unknown'
}

/** Local evidence may verify; web-only, unknown-only, and batch recall stays pending. */
export function placeMatchIsCandidateOnly(match: PlaceMatch | null | undefined): boolean {
  if (match?.candidate_only === true) return true
  if (match?.match_kind === 'geocoder_candidate') return true
  if (match?.match_kind === 'web_candidate') return true
  if (match?.match_kind === 'geocoder_locality') return true
  const sources = normalizePlaceSources(match?.sources)
  if (!sources.length) return false
  return sources.every((source) => placeSourceGroup(source) !== 'local')
}

export function placeMatchVerificationLabel(match: PlaceMatch | null | undefined): string {
  if (!placeMatchIsCandidateOnly(match)) return '地名核验'
  const networkKind = ['web_candidate', 'geocoder_candidate', 'geocoder_locality'].includes(
    match?.match_kind ?? '',
  )
  const networkSource = normalizePlaceSources(match?.sources).some(
    (source) => placeSourceGroup(source) === 'web',
  )
  return networkKind || networkSource ? '联网候选 · 待验证' : '地点候选 · 待验证'
}

export function observationSourceLabel(source: PlaceSource): string {
  const observation = knownObservationSource(source.observation_source)
  return observation ? OBSERVATION_SOURCE_LABELS[observation] : ''
}

export function legacyPlaceSourceLabel(value: unknown): string {
  const text = cleanText(value)
  if (!text) return ''
  const sourceType = knownSourceType(text)
  if (sourceType) return `检索来源：${SOURCE_TYPE_LABELS[sourceType]}`
  const observation = knownObservationSource(text)
  if (observation) return `文字来源：${OBSERVATION_SOURCE_LABELS[observation]}`
  if (safeHttpSourceUrl(text)) return '检索来源：联网来源'
  if (/^[a-z][a-z0-9_.:-]*$/i.test(text)) return '来源待核实'
  return `来源：${boundedSourceText(text, 160)}`
}

export function formatSourceFetchedAt(value: unknown): string {
  const text = boundedSourceText(value, 80)
  if (!text) return ''
  const timestamp = Date.parse(text)
  return Number.isNaN(timestamp)
    ? text
    : new Date(timestamp).toLocaleString('zh-CN', { hour12: false })
}

export interface PlaceSourceDisplay {
  key: string
  title: string
  label: string
  excerpt: string
  fetchedAt: string
  url: string | null
  retrievalLabel: string
  observationLabel: string
}

export function placeSourceDisplay(source: PlaceSource): PlaceSourceDisplay | null {
  const normalized = normalizePlaceSource(source)
  if (!normalized) return null
  return {
    key: placeSourceIdentity(normalized),
    title: boundedSourceText(normalized.title, 140),
    label: boundedSourceText(normalized.label, 100),
    excerpt: boundedSourceText(normalized.excerpt, 320),
    fetchedAt: formatSourceFetchedAt(normalized.fetched_at),
    url: safeHttpSourceUrl(normalized.url),
    retrievalLabel: placeSourceTypeLabel(normalized),
    observationLabel: observationSourceLabel(normalized),
  }
}

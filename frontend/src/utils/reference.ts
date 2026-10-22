import type { CountryScore, ReferenceMatch, ReferenceRegion } from '@/types'

/** UI-facing disclaimer: labeled-reference similarity is retrieval-only diagnostics. */
export const REFERENCE_MATCH_DISCLAIMER = '参考图地点，不是照片定位或 GPS'
export const REFERENCE_SIMILARITY_SEMANTICS =
  '相似度只用于参考图排序，不是地点概率或置信度。'

export const REFERENCE_MATCH_LIMIT = 5

export interface ReferenceCosineRow {
  iso: string
  name_zh: string
  cosine: number
}

function finiteCosine(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** Format similarity evidence explicitly as a score, never as a probability/percentage. */
export function formatReferenceCosine(value: number | null | undefined): string | null {
  const cosine = finiteCosine(value)
  return cosine == null ? null : `相似度 ${cosine.toFixed(3)}（不是概率）`
}

function compareByCosineThenKey<T>(
  a: T,
  b: T,
  cosineOf: (item: T) => number,
  keyOf: (item: T) => string,
): number {
  const delta = cosineOf(b) - cosineOf(a)
  if (delta !== 0) return delta
  const ka = keyOf(a)
  const kb = keyOf(b)
  return ka < kb ? -1 : ka > kb ? 1 : 0
}

/**
 * Flatten and deduplicate IMAGE reference matches across all scored countries.
 * Rows without a usable ``reference_id`` / finite ``raw_cosine`` are dropped so
 * missing backend diagnostics never surface as fabricated matches.
 */
export function topReferenceMatches(
  scores: readonly CountryScore[],
  limit: number = REFERENCE_MATCH_LIMIT,
): ReferenceMatch[] {
  const best = new Map<string, ReferenceMatch>()
  for (const score of scores) {
    for (const match of score.reference_matches ?? []) {
      const id = typeof match?.reference_id === 'string' ? match.reference_id.trim() : ''
      const cosine = finiteCosine(match?.raw_cosine)
      if (!id || cosine == null) continue
      const current = best.get(id)
      if (!current || finiteCosine(current.raw_cosine)! < cosine) best.set(id, match)
    }
  }
  return [...best.values()]
    .sort((a, b) =>
      compareByCosineThenKey(a, b, (item) => finiteCosine(item.raw_cosine) ?? 0, (item) => item.reference_id),
    )
    .slice(0, Math.max(0, limit))
}

/** Flatten region/prototype diagnostics across scores, deduplicated by ISO + region. */
export function collectReferenceRegions(
  scores: readonly CountryScore[],
  limit = REFERENCE_MATCH_LIMIT,
): ReferenceRegion[] {
  const best = new Map<string, ReferenceRegion>()
  for (const score of scores) {
    for (const region of score.reference_regions ?? []) {
      const iso = typeof region?.iso === 'string' ? region.iso.trim() : ''
      const name = typeof region?.region === 'string' ? region.region.trim() : ''
      const cosine = finiteCosine(region?.raw_cosine)
      if (!iso || !name || cosine == null) continue
      const key = `${iso}\u0000${name}`
      const current = best.get(key)
      if (!current || finiteCosine(current.raw_cosine)! < cosine) best.set(key, region)
    }
  }
  return [...best.values()]
    .sort((a, b) =>
      compareByCosineThenKey(
        a,
        b,
        (item) => finiteCosine(item.raw_cosine) ?? 0,
        (item) => `${item.iso}\u0000${item.region}`,
      ),
    )
    .slice(0, Math.max(0, limit))
}

/** Per-country scalar reference cosine, used when region rows are absent. */
export function referenceCosineRows(scores: readonly CountryScore[]): ReferenceCosineRow[] {
  const rows: ReferenceCosineRow[] = []
  for (const score of scores) {
    const cosine = finiteCosine(score.reference_cosine)
    if (cosine == null) continue
    rows.push({ iso: score.iso, name_zh: score.name_zh || score.iso, cosine })
  }
  return rows.sort((a, b) =>
    compareByCosineThenKey(a, b, (item) => item.cosine, (item) => item.iso),
  )
}

function hasOwn<T extends object>(value: T, key: keyof T | string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

/**
 * True when the backend included reference diagnostics at all. An empty array
 * means "library configured but no matches", while absent fields mean no
 * labeled reference library is configured.
 */
export function hasReferenceField(scores: readonly CountryScore[]): boolean {
  return scores.some(
    (score) =>
      hasOwn(score, 'reference_matches') ||
      hasOwn(score, 'reference_regions') ||
      hasOwn(score, 'reference_cosine'),
  )
}

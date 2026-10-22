import type { CityScore } from '@/types'

export interface CityDisplayNameSource {
  name_zh?: string | null
  name?: string | null
  label?: string | null
}

function cleanCityName(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function displayableCityName(value: unknown): string {
  const name = cleanCityName(value)
  // Old persisted sessions may still carry the removed placeholder.
  return name.includes('中文名待补充') ? '' : name
}

/**
 * One title policy for city rows and guided city targets. Prefer the
 * backend-localized name, then the authoritative original name; only render an
 * unnamed marker when no usable name exists at all.
 */
export function displayCityName(
  city: CityDisplayNameSource | null | undefined,
): string {
  const localized = displayableCityName(city?.name_zh)
  if (localized) return localized
  return displayableCityName(city?.name) || displayableCityName(city?.label) || '未命名城市'
}

/**
 * City rows must always expose image similarity, while any combined sort score
 * is labelled honestly as a non-probabilistic rank.
 */
export function cityScoreLabel(city: CityScore): string {
  const cosine = Number.isFinite(city.raw_cosine) ? city.raw_cosine.toFixed(3) : '—'
  const composite = Number.isFinite(city.score) ? city.score.toFixed(3) : '—'
  if (city.score_type === 'city_clue_fusion') {
    return `综合匹配 ${composite} · 图片相似度 ${cosine}（不是概率）`
  }
  if (city.score_type === 'city_reference_hybrid') {
    return `综合匹配 ${composite} · 图片相似度 ${cosine}（不是概率）`
  }
  return `图片相似度 ${cosine}（不是概率）`
}

/** Number of clue-matching diagnostics attached to a fused city row. */
export function cityMatchedClueCount(city: CityScore): number {
  return Array.isArray(city.matched_clues) ? city.matched_clues.length : 0
}

import type { CityScore } from '@/types'

export interface CityDisplayNameSource {
  name_zh?: string | null
  name?: string | null
  label?: string | null
}

function cleanCityName(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function isChineseCityName(value: string): boolean {
  // Keep genuine Chinese originals (including common separators), but never
  // infer a translation from a mixed or Latin-script city name.
  const compact = value.replace(/[\s·・,，.。'’"“”()（）【】\[\]\-—_\/]+/g, '')
  return compact.length > 0 && /^[\u3400-\u9fff\uf900-\ufaff]+$/.test(compact)
}

/**
 * One Chinese title policy for city rows and guided city targets. Prefer the
 * backend-localized name, then preserve an already-Chinese original; otherwise
 * show an explicit gap instead of guessing from an English exonym.
 */
export function displayCityName(
  city: CityDisplayNameSource | null | undefined,
): string {
  const localized = cleanCityName(city?.name_zh)
  if (localized) return localized
  const chineseOriginal = [cleanCityName(city?.name), cleanCityName(city?.label)].find(
    isChineseCityName,
  )
  return chineseOriginal || '中文名待补充'
}

/**
 * City rows must always expose the raw full-image cosine, while any combined
 * sort score is labelled honestly as a non-probabilistic rank.
 */
export function cityScoreLabel(city: CityScore): string {
  const cosine = Number.isFinite(city.raw_cosine) ? city.raw_cosine.toFixed(3) : '—'
  const composite = Number.isFinite(city.score) ? city.score.toFixed(3) : '—'
  if (city.score_type === 'city_clue_fusion') {
    return `线索融合排序 ${composite} · 全图余弦 ${cosine}（非概率）`
  }
  if (city.score_type === 'city_reference_hybrid') {
    return `融合排序 ${composite} · 全图余弦 ${cosine}（非概率）`
  }
  return `城市文本余弦 ${cosine}（非概率）`
}

/** Number of clue-matching diagnostics attached to a fused city row. */
export function cityMatchedClueCount(city: CityScore): number {
  return Array.isArray(city.matched_clues) ? city.matched_clues.length : 0
}
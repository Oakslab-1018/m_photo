import type { CityScore } from '@/types'

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
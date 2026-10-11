import type { CountryScore } from '@/types'

export interface PercentPosition {
  x: number
  y: number
}

export const MAP_LON_MIN = -180
export const MAP_LON_MAX = 180
export const MAP_LAT_MIN = -60
export const MAP_LAT_MAX = 80

function clamp(value: number, min = 0, max = 100): number {
  return Math.min(max, Math.max(min, value))
}

/** Project lat/lng into the original 1600x622 world map's percentage box. */
export function latLngToPercent(lat: number, lng: number): PercentPosition {
  const safeLat = Number.isFinite(lat) ? lat : 0
  const safeLng = Number.isFinite(lng) ? lng : 0
  const lonSpan = MAP_LON_MAX - MAP_LON_MIN
  const latSpan = MAP_LAT_MAX - MAP_LAT_MIN
  const x = ((safeLng - MAP_LON_MIN) / (lonSpan || 1)) * 100
  const y = ((MAP_LAT_MAX - safeLat) / (latSpan || 1)) * 100
  return { x: clamp(x), y: clamp(y) }
}

export function markerSize(score: number): number {
  const safe = Number.isFinite(score) ? Math.min(1, Math.max(0, score)) : 0
  return 16 + safe * 10
}

export function markerColor(rank: number): string {
  if (rank === 0) return '#22c55e'
  if (rank === 1) return '#f59e0b'
  return '#94a3b8'
}

export function scorePercent(score: number): string {
  const safe = Number.isFinite(score) ? Math.min(1, Math.max(0, score)) : 0
  return `${(safe * 100).toFixed(1)}%`
}

export function sortScores(scores: readonly CountryScore[]): CountryScore[] {
  return [...scores].sort((a, b) => safeScore(b) - safeScore(a))
}

function safeScore(score: CountryScore): number {
  return Number.isFinite(score.score) ? score.score : 0
}

/** Ordering evidence, never a calibrated geolocation probability. */
export function scoreLabel(score: CountryScore): string {
  const rawCosine =
    score.raw_cosine != null && Number.isFinite(score.raw_cosine) ? score.raw_cosine : null
  const composite = Number.isFinite(score.score) ? score.score : 0

  if (score.is_grouped || score.fusion?.aggregation === 'max') {
    return rawCosine != null
      ? `地区汇总排序 ${composite.toFixed(3)} · 来源余弦 ${rawCosine.toFixed(3)}（非概率）`
      : `地区汇总权重 ${composite.toFixed(3)}（非概率）`
  }
  if (score.score_type === 'hybrid_rank') {
    // Composite fusion score must never be presented as a lone cosine/probability.
    return rawCosine != null
      ? `融合排序 ${composite.toFixed(3)} · 全图余弦 ${rawCosine.toFixed(3)}（非概率）`
      : `融合排序 ${composite.toFixed(3)}（非概率）`
  }

  if (rawCosine != null) {
    return `余弦相似度 ${rawCosine.toFixed(3)}`
  }
  return `规则权重 ${composite.toFixed(3)}`
}

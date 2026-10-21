import type { Clue, RegionBBox } from '@/types'

/**
 * Backend coordinates are normalized to the original image. Keep the same
 * tiny floating-point allowance as the backend validator without clamping or
 * inventing geometry.
 */
const BOUND_EPSILON = 1e-9

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finiteBox(value: unknown): RegionBBox | null {
  if (!isRecord(value)) return null
  const { x, y, w, h } = value
  if (
    typeof x !== 'number'
    || typeof y !== 'number'
    || typeof w !== 'number'
    || typeof h !== 'number'
    || !Number.isFinite(x)
    || !Number.isFinite(y)
    || !Number.isFinite(w)
    || !Number.isFinite(h)
  ) {
    return null
  }
  if (x < 0 || y < 0 || w <= 0 || h <= 0) return null
  if (x + w > 1 + BOUND_EPSILON || y + h > 1 + BOUND_EPSILON) return null
  return { x, y, w, h }
}

/**
 * Return the real source box attached by region analysis, preferring the
 * individual clue box and falling back to the analyzed source region.
 *
 * This intentionally reads only backend geometry. It never derives a box from
 * clue text or value.
 */
export function clueRegionBox(clue: Clue | null | undefined): RegionBBox | null {
  const extra = clue?.extra
  if (!isRecord(extra)) return null
  return finiteBox(extra.bbox_norm) ?? finiteBox(extra.region_bbox_norm)
}

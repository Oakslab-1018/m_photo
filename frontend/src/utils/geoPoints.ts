import type { CityScore, CountryScore } from '@/types'
import { cityRegionLabel, regionScopeLabel } from '@/utils/geo'

export interface GeoPoint {
  key: string
  kind: 'country' | 'city'
  iso: string
  name: string
  lat: number
  lng: number
  cosine: number | null
  scoreType?: string
  confirmed: boolean
  /** Optional region scope (中国台湾 / 中国香港 / 中国澳门) for tooltip evidence. */
  subregion?: string | null
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** Validates a lng/lat pair: finite, in range, and not the null-island sentinel. */
export function isUsableLngLat(lng: unknown, lat: unknown): boolean {
  const lon = finite(lng)
  const la = finite(lat)
  if (lon == null || la == null) return false
  if (lon < -180 || lon > 180 || la < -90 || la > 90) return false
  if (Math.abs(lon) < 1e-9 && Math.abs(la) < 1e-9) return false
  return true
}

/** Country markers from ranked scores; invalid or zero coordinates are skipped. */
export function countryPoints(
  scores: readonly CountryScore[],
  confirmedIso: string | null,
): GeoPoint[] {
  const points: GeoPoint[] = []
  for (const score of scores) {
    if (!isUsableLngLat(score.lng, score.lat)) continue
    points.push({
      key: `country:${score.iso}`,
      kind: 'country',
      iso: score.iso,
      name: score.name_zh || score.name || score.iso,
      lat: score.lat,
      lng: score.lng,
      cosine: finite(score.raw_cosine),
      scoreType: score.score_type,
      confirmed: confirmedIso === score.iso,
      subregion: regionScopeLabel(score),
    })
  }
  return points
}

/** City candidate markers: representative city points, never photo GPS. */
export function cityPoints(cities: readonly CityScore[] | undefined): GeoPoint[] {
  const points: GeoPoint[] = []
  const seen = new Set<string>()
  for (const city of cities ?? []) {
    if (!city || typeof city.city_id !== 'string' || !city.city_id.trim()) continue
    if (seen.has(city.city_id)) continue
    if (!isUsableLngLat(city.lng, city.lat)) continue
    seen.add(city.city_id)
    points.push({
      key: `city:${city.city_id}`,
      kind: 'city',
      iso: city.iso,
      name: city.name || city.city_id,
      lat: city.lat,
      lng: city.lng,
      cosine: finite(city.raw_cosine),
      scoreType: city.score_type,
      confirmed: false,
      subregion: cityRegionLabel(city),
    })
  }
  return points
}

export interface LngLatBoundsLike {
  west: number
  south: number
  east: number
  north: number
}

/**
 * Bounding box over points, dateline-aware: longitudes are unwrapped into a
 * continuous range (e.g. 175 and -175 -> 175..185) instead of spanning the
 * whole globe. Returns null when no usable point exists.
 *
 * The window is the complement of the *widest* gap between adjacent longitudes:
 * every other gap lies inside the window, so no point falls outside it. Picking
 * the narrowest gap instead would frame a pair of neighbours and silently drop
 * the rest — e.g. [2.2, 8.5, 12.5] would fit 8.5..12.5 and cut off the
 * top-ranked candidate at 2.2.
 */
export function boundsForPoints(
  points: readonly { lat: number; lng: number }[],
): LngLatBoundsLike | null {
  if (!points.length) return null
  const lats = points.map((point) => point.lat)
  const lngs = points.map((point) => point.lng).sort((a, b) => a - b)
  const south = Math.min(...lats)
  const north = Math.max(...lats)

  let widest = -1
  let west = lngs[0]
  let east = lngs[lngs.length - 1]
  for (let index = 0; index < lngs.length; index += 1) {
    const start = lngs[index]
    const wrapped = index === lngs.length - 1
    const end = wrapped ? lngs[0] + 360 : lngs[index + 1]
    if (end - start > widest) {
      widest = end - start
      // Fitting on the far side of the widest gap keeps every point inside.
      west = wrapped ? lngs[0] : lngs[index + 1]
      east = wrapped ? lngs[lngs.length - 1] : lngs[index] + 360
    }
  }
  return { west, south, east, north }
}

/** Overlay stack used by the interactive map. Country markers sit on top. */
export function markerStack(
  countries: readonly GeoPoint[],
  cities: readonly GeoPoint[],
): GeoPoint[] {
  return [...cities, ...countries]
}

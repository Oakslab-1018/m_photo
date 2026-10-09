import { describe, expect, it } from 'vitest'
import {
  MAPLIBRE_DEMO_STYLE_URL,
  resolveMapStyle,
} from '@/utils/mapConfig'
import {
  boundsForPoints,
  cityPoints,
  countryPoints,
  isUsableLngLat,
  markerStack,
} from '@/utils/geoPoints'
import type { CityScore, CountryScore } from '@/types'

function country(overrides: Partial<CountryScore> = {}): CountryScore {
  return { iso: 'JP', name: 'Japan', name_zh: '日本', score: 0.5, lat: 36.2, lng: 138.25, ...overrides }
}

function city(overrides: Partial<CityScore> = {}): CityScore {
  return {
    city_id: 'jp-tokyo',
    iso: 'JP',
    name: '东京',
    region: '关东',
    lat: 35.68,
    lng: 139.69,
    raw_cosine: 0.7,
    score: 0.7,
    score_type: 'city_text_cosine',
    ...overrides,
  }
}

describe('resolveMapStyle', () => {
  it('prefers an explicit style URL', () => {
    const config = resolveMapStyle({
      VITE_MAP_STYLE_URL: 'https://tiles.example.com/style.json',
      VITE_MAPTILER_API_KEY: 'abc',
    })
    expect(config).toEqual({
      styleUrl: 'https://tiles.example.com/style.json',
      source: 'configured',
      demoOnly: false,
    })
  })

  it('builds a MapTiler style URL from a public key', () => {
    const config = resolveMapStyle({ VITE_MAPTILER_API_KEY: 'key with space' })
    expect(config.source).toBe('configured')
    expect(config.demoOnly).toBe(false)
    expect(config.styleUrl).toBe(
      'https://api.maptiler.com/maps/streets-v4/style.json?key=key%20with%20space',
    )
  })

  it('falls back to demo tiles and flags them as demo-only', () => {
    for (const env of [{}, { VITE_MAP_STYLE_URL: '   ' }, { VITE_MAPTILER_API_KEY: '' }]) {
      const config = resolveMapStyle(env)
      expect(config.styleUrl).toBe(MAPLIBRE_DEMO_STYLE_URL)
      expect(config.source).toBe('demo')
      expect(config.demoOnly).toBe(true)
    }
  })
})

describe('isUsableLngLat', () => {
  it('accepts valid coordinates and rejects out-of-range, non-finite, and null-island', () => {
    expect(isUsableLngLat(139.69, 35.68)).toBe(true)
    expect(isUsableLngLat(-180, -90)).toBe(true)
    expect(isUsableLngLat(181, 0)).toBe(false)
    expect(isUsableLngLat(0, 91)).toBe(false)
    expect(isUsableLngLat(Number.NaN, 0)).toBe(false)
    expect(isUsableLngLat(0, Number.POSITIVE_INFINITY)).toBe(false)
    expect(isUsableLngLat(0, 0)).toBe(false)
  })
})

describe('countryPoints / cityPoints', () => {
  it('filters unusable country coordinates and marks confirmation', () => {
    const points = countryPoints(
      [
        country(),
        country({ iso: 'US', name_zh: '美国', lat: 0, lng: 0 }),
        country({ iso: 'CA', name_zh: '加拿大', lat: 999, lng: -106 }),
      ],
      'JP',
    )
    expect(points.map((point) => point.key)).toEqual(['country:JP'])
    expect(points[0].confirmed).toBe(true)
    expect(points[0].cosine).toBeNull()
  })

  it('deduplicates cities by id and drops invalid/missing data', () => {
    const points = cityPoints([
      city(),
      city({ name: '重复东京' }),
      city({ city_id: '' }),
      city({ city_id: 'bad', lat: 0, lng: 0 }),
      city({ city_id: 'nocoord', lat: Number.NaN }),
    ])
    expect(points).toHaveLength(1)
    expect(points[0].name).toBe('东京')
  })

  it('puts country markers above city markers in the overlay stack', () => {
    const stack = markerStack(countryPoints([country()], null), cityPoints([city()]))
    expect(stack.map((point) => point.kind)).toEqual(['city', 'country'])
  })
})

describe('boundsForPoints', () => {
  it('returns null when there is no usable point', () => {
    expect(boundsForPoints([])).toBeNull()
  })

  it('wraps around the dateline instead of spanning the globe', () => {
    const bounds = boundsForPoints([
      { key: 'a', kind: 'city', iso: 'FJ', name: 'A', lat: -18, lng: 175, cosine: null, confirmed: false },
      { key: 'b', kind: 'city', iso: 'NZ', name: 'B', lat: -41, lng: -175, cosine: null, confirmed: false },
    ])
    expect(bounds).not.toBeNull()
    expect(bounds!.west).toBe(175)
    expect(bounds!.east).toBe(185)
    expect(bounds!.east - bounds!.west).toBeLessThan(20)
  })

  it('fits ordinary spread of points', () => {
    const bounds = boundsForPoints(countryPoints([country(), country({ iso: 'FR', lat: 46.2, lng: 2.2 })], null))
    expect(bounds!.west).toBe(2.2)
    expect(bounds!.east).toBe(138.25)
    expect(bounds!.south).toBe(36.2)
    expect(bounds!.north).toBe(46.2)
  })
})

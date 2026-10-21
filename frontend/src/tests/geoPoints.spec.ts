import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  ESRI_WORLD_IMAGERY_STYLE_URL,
  resolveMapStyle,
} from '@/utils/mapConfig'
import {
  boundsForPoints,
  cityPoints,
  countryPoints,
  guidedLocationPoints,
  isUsableLngLat,
  markerStack,
} from '@/utils/geoPoints'
import type {
  CityScore,
  CountryScore,
  GuidedConfirmedPath,
  LocationCandidate,
  LocationTarget,
} from '@/types'

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

  it('defaults to bundled Esri World Imagery without marking it demo-only', () => {
    for (const env of [{}, { VITE_MAP_STYLE_URL: '   ' }, { VITE_MAPTILER_API_KEY: '' }]) {
      const config = resolveMapStyle(env)
      expect(config.styleUrl).toBe(`${import.meta.env.BASE_URL}esri-world-imagery-style.json`)
      expect(config.styleUrl).toBe(ESRI_WORLD_IMAGERY_STYLE_URL)
      expect(config.source).toBe('default')
      expect(config.demoOnly).toBe(false)
    }
  })

  it('bundles the exact Esri World Imagery raster contract', () => {
    const style = JSON.parse(
      readFileSync(new URL('../../public/esri-world-imagery-style.json', import.meta.url), 'utf8'),
    )
    expect(style.sources['esri-world-imagery']).toMatchObject({
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      maxzoom: 23,
      attribution: 'Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community',
    })
    expect(style.layers).toContainEqual(
      expect.objectContaining({ type: 'raster', source: 'esri-world-imagery' }),
    )
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

describe('guidedLocationPoints', () => {
  it('ignores deprecated admin1 path metadata while preserving city provenance', () => {
    const japan: LocationTarget = {
      level: 'country',
      id: 'JP',
      label: '日本',
      country_iso: 'JP',
      lat: 36.2,
      lng: 138.25,
    }
    const legacyAdmin1: LocationTarget = {
      level: 'admin1',
      id: 'JP.01',
      label: '北海道',
      country_iso: 'JP',
      admin1_id: 'JP.01',
      lat: 43.06,
      lng: 141.35,
    }
    const sapporo: LocationCandidate = {
      target: {
        level: 'city',
        id: '1001',
        label: '札幌',
        country_iso: 'JP',
        admin1_id: 'JP.01',
        city_id: '1001',
        lat: 43.06,
        lng: 141.35,
      },
      score: 0.7,
    }
    const path = {
      country: japan,
      admin1: legacyAdmin1,
      city: null,
    } satisfies GuidedConfirmedPath

    const result = guidedLocationPoints(
      [sapporo, { target: legacyAdmin1, score: 0.9 }],
      path,
      'city',
    )

    expect(result.points.map((point) => point.key)).toEqual(['country:JP', 'city:1001'])
    expect(result.points.some((point) => point.level === 'admin1')).toBe(false)
    expect(result.fallbackPoints.map((point) => point.key)).toEqual(['country:JP'])
    expect(result.missingCoordinates).toEqual([])
    expect(result.points.find((point) => point.level === 'city')?.target?.admin1_id).toBe('JP.01')
  })
})

describe('boundsForPoints', () => {
  it('returns null when there is no usable point', () => {
    expect(boundsForPoints([])).toBeNull()
  })

  it('wraps around the dateline instead of spanning the globe', () => {
    const bounds = boundsForPoints([
      { lat: -18, lng: 175 },
      { lat: -41, lng: -175 },
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

  it('keeps a lone outlier inside the window instead of framing the closest pair', () => {
    // 2.2 / 8.5 / 12.5: the tightest gap is 8.5..12.5, which would drop the
    // westernmost candidate off the map entirely.
    const bounds = boundsForPoints([
      { lat: 46.2, lng: 2.2 },
      { lat: 46.8, lng: 8.5 },
      { lat: 47.3, lng: 12.5 },
    ])
    expect(bounds!.west).toBe(2.2)
    expect(bounds!.east).toBe(12.5)
  })

  it('spans every point across a widely spread candidate set', () => {
    const bounds = boundsForPoints([
      { lat: 36.2, lng: 138.25 },
      { lat: 23.7, lng: 121 },
      { lat: 36.5, lng: 127.5 },
      { lat: 7.5, lng: 134.5 },
      { lat: -21.1, lng: 55.5 },
    ])
    expect(bounds!.west).toBe(55.5)
    expect(bounds!.east).toBe(138.25)
  })
})

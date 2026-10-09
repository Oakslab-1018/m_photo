import { describe, expect, it } from 'vitest'
import { latLngToPercent, markerSize, scorePercent, scoreLabel } from '@/utils/map'
import { toMapMarkers } from '@/composables/useMapViewport'
import type { CountryScore } from '@/types'

describe('map coordinate projection', () => {
  it('maps the exact -180..180 / -60..80 bounds to 0..100', () => {
    expect(latLngToPercent(-60, -180)).toEqual({ x: 0, y: 100 })
    expect(latLngToPercent(80, 180)).toEqual({ x: 100, y: 0 })
    expect(latLngToPercent(0, 0)).toEqual({ x: 50, y: 57.14285714285714 })
  })

  it('clamps out-of-range coordinates', () => {
    expect(latLngToPercent(200, 360)).toEqual({ x: 100, y: 0 })
    expect(latLngToPercent(-200, -360)).toEqual({ x: 0, y: 100 })
  })

  it('formats scores and marker sizes safely', () => {
    expect(scorePercent(0.856)).toBe('85.6%')
    expect(scorePercent(2)).toBe('100.0%')
    expect(markerSize(1)).toBe(26)
    expect(markerSize(Number.NaN)).toBe(16)
  })
})

describe('country marker hemispheres', () => {
  function country(iso: string, lat: number, lng: number, name_zh = iso): CountryScore {
    return { iso, name: iso, name_zh, score: 0.5, lat, lng }
  }

  it('places western-hemisphere countries left of the prime meridian', () => {
    const west = [
      country('US', 37.0902, -95.7129),
      country('CA', 56.1304, -106.3468),
      country('MX', 23.6345, -102.5528),
      country('BR', -14.235, -51.9253),
      country('GB', 55.3781, -3.436),
      country('ES', 40.4637, -3.7492),
    ]
    for (const item of west) {
      const { x } = latLngToPercent(item.lat, item.lng)
      expect(x, `${item.iso} should be west of 50%`).toBeLessThan(50)
    }
  })

  it('places eastern-hemisphere countries right of the prime meridian', () => {
    const east = [
      country('JP', 36.2048, 138.2529),
      country('CN', 35.8617, 104.1954),
      country('KR', 35.9078, 127.7669),
      country('AU', -25.2744, 133.7751),
      country('IN', 20.5937, 78.9629),
    ]
    for (const item of east) {
      const { x } = latLngToPercent(item.lat, item.lng)
      expect(x, `${item.iso} should be east of 50%`).toBeGreaterThan(50)
    }
  })

  it('places southern-hemisphere countries below the equator', () => {
    for (const item of [country('BR', -14.235, -51.9253), country('AU', -25.2744, 133.7751)]) {
      const { y } = latLngToPercent(item.lat, item.lng)
      expect(y, `${item.iso} should be south of the equator`).toBeGreaterThan(50)
    }
  })

  it('describes markers as candidates, and confirmed ones only as countries', () => {
    const scores = [country('GB', 55.3781, -3.436, '英国')]
    const [candidate] = toMapMarkers(scores, null)
    expect(candidate.title).toContain('候选国家')
    expect(candidate.title).not.toContain('已确认拍摄地')

    const [confirmed] = toMapMarkers(scores, 'GB')
    expect(confirmed.title).toContain('用户已确认的国家')
    expect(confirmed.title).toContain('非照片 GPS')
    expect(confirmed.title).not.toContain('已确认拍摄地')
  })
})

describe('non-probabilistic evidence display', () => {
  const base = { iso: 'JP', name: 'Japan', name_zh: '日本', score: 1, lat: 0, lng: 0 }
  it('shows raw cosine, not an artificial 100 percent', () => {
    expect(scoreLabel({ ...base, raw_cosine: 0.27432 })).toBe('余弦相似度 0.274')
    expect(scoreLabel(base)).toBe('规则权重 1.000')
  })

  it('shows composite and full-image cosine distinctly for hybrid_rank', () => {
    const hybrid = {
      ...base,
      score: 0.412,
      raw_cosine: 0.274,
      score_type: 'hybrid_rank',
      fusion: { streetclip: 0.274, rules: 0.138 },
    }
    const label = scoreLabel(hybrid)
    expect(label).toContain('融合排序 0.412')
    expect(label).toContain('全图余弦 0.274')
    expect(label).toContain('非概率')
    expect(label).not.toContain('%')
    expect(label).not.toBe(`余弦相似度 ${hybrid.raw_cosine.toFixed(3)}`)
  })
})

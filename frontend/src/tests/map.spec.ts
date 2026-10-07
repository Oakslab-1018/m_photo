import { describe, expect, it } from 'vitest'
import { latLngToPercent, markerSize, scorePercent } from '@/utils/map'

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

import { describe, expect, it } from 'vitest'
import type { CityScore } from '@/types'
import { cityMatchedClueCount, cityScoreLabel } from '@/utils/city'

function city(overrides: Partial<CityScore> = {}): CityScore {
  return {
    city_id: 'jp-tokyo',
    iso: 'JP',
    name: '东京',
    region: '关东',
    lat: 35.68,
    lng: 139.69,
    raw_cosine: 0.744,
    score: 0.812,
    score_type: 'city_text_cosine',
    ...overrides,
  }
}

describe('city score labels', () => {
  it('labels clue-fused rows as a combined sort while preserving the raw cosine', () => {
    const label = cityScoreLabel(
      city({ score_type: 'city_clue_fusion', matched_clues: [{}, {}], fusion: { clue_weight: 0.3 } }),
    )
    expect(label).toContain('线索融合排序 0.812')
    expect(label).toContain('全图余弦 0.744')
    expect(label).toContain('非概率')
  })

  it('keeps reference hybrid and plain cosine labels honest', () => {
    expect(cityScoreLabel(city({ score_type: 'city_reference_hybrid' }))).toContain('融合排序 0.812')
    expect(cityScoreLabel(city())).toBe('城市文本余弦 0.744（非概率）')
  })

  it('counts matched clue diagnostics and tolerates missing data', () => {
    expect(cityMatchedClueCount(city({ matched_clues: [{}, {}, {}] }))).toBe(3)
    expect(cityMatchedClueCount(city())).toBe(0)
  })
})
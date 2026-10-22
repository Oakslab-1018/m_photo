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
    expect(label).toContain('综合匹配 0.812')
    expect(label).toContain('图片相似度 0.744')
    expect(label).toContain('不是概率')
  })

  it('keeps reference hybrid and plain similarity labels honest', () => {
    expect(cityScoreLabel(city({ score_type: 'city_reference_hybrid' }))).toContain('综合匹配 0.812')
    expect(cityScoreLabel(city())).toBe('图片相似度 0.744（不是概率）')
  })

  it('counts matched clue diagnostics and tolerates missing data', () => {
    expect(cityMatchedClueCount(city({ matched_clues: [{}, {}, {}] }))).toBe(3)
    expect(cityMatchedClueCount(city())).toBe(0)
  })
})
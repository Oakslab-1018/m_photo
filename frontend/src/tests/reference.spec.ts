import { describe, expect, it } from 'vitest'
import type { CountryScore } from '@/types'
import {
  REFERENCE_MATCH_DISCLAIMER,
  REFERENCE_SIMILARITY_SEMANTICS,
  collectReferenceRegions,
  formatReferenceCosine,
  hasReferenceField,
  referenceCosineRows,
  topReferenceMatches,
} from '@/utils/reference'

function score(overrides: Partial<CountryScore> = {}): CountryScore {
  return {
    iso: 'JP',
    name: 'Japan',
    name_zh: '日本',
    score: 0.4,
    raw_cosine: 0.3,
    lat: 36.2,
    lng: 138.25,
    ...overrides,
  }
}

describe('topReferenceMatches', () => {
  it('returns deduplicated top 5 IMAGE matches sorted by raw cosine', () => {
    const scores = [
      score({
        reference_matches: [
          { reference_id: 'r1', iso: 'JP', region: 'Kanto', place: 'Shibuya', raw_cosine: 0.91 },
          { reference_id: 'r2', iso: 'JP', region: 'Kansai', place: 'Osaka', raw_cosine: 0.72 },
          { reference_id: 'r3', iso: 'KR', region: 'Seoul', place: 'Gangnam', raw_cosine: 0.65 },
        ],
      }),
      score({
        iso: 'KR',
        reference_matches: [
          // Duplicate id with a lower cosine must not displace the higher one.
          { reference_id: 'r1', iso: 'JP', region: 'Kanto', place: 'Shibuya', raw_cosine: 0.11 },
          { reference_id: 'r4', iso: 'KR', region: 'Seoul', place: 'Hongdae', raw_cosine: 0.6 },
          { reference_id: 'r5', iso: 'TH', region: 'Bangkok', place: 'Sukhumvit', raw_cosine: 0.55 },
          { reference_id: 'r6', iso: 'IN', region: 'Delhi', place: 'Connaught', raw_cosine: 0.5 },
          { reference_id: 'r7', iso: 'EG', region: 'Cairo', place: 'Zamalek', raw_cosine: 0.45 },
        ],
      }),
    ]

    const matches = topReferenceMatches(scores)
    expect(matches).toHaveLength(5)
    expect(matches.map((match) => match.reference_id)).toEqual(['r1', 'r2', 'r3', 'r4', 'r5'])
    expect(matches[0].raw_cosine).toBe(0.91)
  })

  it('uses deterministic reference_id ordering for equal cosines', () => {
    const scores = [
      score({
        reference_matches: [
          { reference_id: 'zeta', iso: 'JP', region: 'A', place: 'Z', raw_cosine: 0.5 },
          { reference_id: 'alpha', iso: 'JP', region: 'A', place: 'A', raw_cosine: 0.5 },
          { reference_id: 'mid', iso: 'JP', region: 'A', place: 'M', raw_cosine: 0.5 },
        ],
      }),
    ]
    expect(topReferenceMatches(scores).map((m) => m.reference_id)).toEqual([
      'alpha',
      'mid',
      'zeta',
    ])
  })

  it('drops malformed or missing data instead of fabricating matches', () => {
    const matches = topReferenceMatches([
      score({
        reference_matches: [
          { reference_id: '', iso: 'JP', region: 'A', place: 'A', raw_cosine: 0.9 },
          { reference_id: 'null-cos', iso: 'JP', region: 'A', place: 'A', raw_cosine: Number.NaN },
          { reference_id: 'ok', iso: 'JP', region: 'A', place: 'A', raw_cosine: 0.4 },
        ],
      }),
      score({ reference_matches: undefined }),
    ])
    expect(matches.map((m) => m.reference_id)).toEqual(['ok'])
  })

  it('never presents cosine similarity as a probability or percentage', () => {
    expect(REFERENCE_MATCH_DISCLAIMER).toContain('不是照片定位或 GPS')
    expect(formatReferenceCosine(0.5)).toBe('相似度 0.500（不是概率）')
    expect(formatReferenceCosine(null)).toBeNull()
    expect(formatReferenceCosine(Number.NaN)).toBeNull()
    expect(REFERENCE_SIMILARITY_SEMANTICS).toContain('不是地点概率');
    expect(REFERENCE_SIMILARITY_SEMANTICS).toContain('置信度')
    expect(REFERENCE_SIMILARITY_SEMANTICS).not.toContain('%')
  })
})

describe('collectReferenceRegions', () => {
  it('deduplicates by iso + region, keeping the best cosine', () => {
    const regions = collectReferenceRegions([
      score({
        reference_regions: [
          { iso: 'JP', region: 'Kanto', raw_cosine: 0.7, prototype_count: 12, support: 8 },
          { iso: 'JP', region: 'Kansai', raw_cosine: 0.6, prototype_count: 6, support: 4 },
        ],
      }),
      score({
        iso: 'JP',
        reference_regions: [
          { iso: 'JP', region: 'Kanto', raw_cosine: 0.82, prototype_count: 20, support: 15 },
          { iso: 'KR', region: 'Seoul', raw_cosine: 0.5, prototype_count: 3, support: 2 },
        ],
      }),
    ])
    expect(regions.map((r) => `${r.iso}:${r.region}`)).toEqual(['JP:Kanto', 'JP:Kansai', 'KR:Seoul'])
    expect(regions[0].prototype_count).toBe(20)
    expect(regions[0].support).toBe(15)
  })

  it('skips rows missing iso, region, or a finite cosine', () => {
    const regions = collectReferenceRegions([
      score({
        reference_regions: [
          { iso: '', region: 'A', raw_cosine: 0.9, prototype_count: 1, support: 1 },
          { iso: 'JP', region: '', raw_cosine: 0.8, prototype_count: 1, support: 1 },
          { iso: 'JP', region: 'B', raw_cosine: Infinity, prototype_count: 1, support: 1 },
          { iso: 'JP', region: 'C', raw_cosine: 0.5, prototype_count: 2, support: 2 },
        ],
      }),
    ])
    expect(regions.map((r) => r.region)).toEqual(['C'])
  })
})

describe('referenceCosineRows / hasReferenceField', () => {
  it('exposes scalar reference cosines per country, sorted with deterministic ties', () => {
    const rows = referenceCosineRows([
      score({ iso: 'JP', name_zh: '日本', reference_cosine: 0.4 }),
      score({ iso: 'KR', name_zh: '韩国', reference_cosine: 0.6 }),
      score({ iso: 'CN', name_zh: '中国', reference_cosine: 0.6 }),
      score({ iso: 'US', name_zh: '美国', reference_cosine: null }),
    ])
    expect(rows.map((r) => r.iso)).toEqual(['CN', 'KR', 'JP'])
  })

  it('distinguishes absent fields (no library) from empty arrays (no matches)', () => {
    expect(hasReferenceField([score()])).toBe(false)
    expect(hasReferenceField([score({ reference_matches: [] })])).toBe(true)
    expect(hasReferenceField([score({ reference_regions: [] })])).toBe(true)
    expect(hasReferenceField([score({ reference_cosine: null })])).toBe(true)
  })
})

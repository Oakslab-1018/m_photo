import { describe, expect, it } from 'vitest'
import type { Clue } from '@/types'
import { confidencePercent, groupClues } from '@/utils/clues'

const clues: Clue[] = [
  { dimension: 'vehicle', value: 'dark_vehicle', confidence: 0.4 },
  { dimension: 'language', value: 'ja', confidence: 0.6 },
  { dimension: 'language', value: 'ja', confidence: 0.9, evidence: '招牌文字' },
  { dimension: 'vegetation', value: 'temperate_forest', confidence: 0.7 },
  { dimension: 'unknown_dimension', value: 'mystery', confidence: 0.5 },
  { dimension: 'language', value: '', confidence: 1 },
]

describe('clue grouping', () => {
  it('groups by dimension, dedupes by value keeping highest confidence, and orders known dimensions', () => {
    const groups = groupClues(clues)
    expect(groups.map((group) => group.dimension)).toEqual([
      'language',
      'vegetation',
      'vehicle',
      'unknown_dimension',
    ])
    expect(groups[0]?.label).toBe('文字语言')
    expect(groups[0]?.items).toHaveLength(1)
    expect(groups[0]?.items[0]?.confidence).toBe(0.9)
  })

  it('formats confidence as a clamped percentage', () => {
    expect(confidencePercent(0.987)).toBe('99%')
    expect(confidencePercent(2)).toBe('100%')
    expect(confidencePercent(Number.NaN)).toBe('0%')
  })
})

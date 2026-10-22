import { describe, expect, it } from 'vitest'
import type { Clue } from '@/types'
import {
  cluePreview,
  clueProvenance,
  clueSourceLabel,
  clueValueLabel,
  confidencePercent,
  countGroupedClues,
  defaultOpenDimensions,
  dimensionLabel,
  groupClues,
  isCatalogValidated,
  isLowConfidence,
  isReferenceOnly,
} from '@/utils/clues'

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

describe('expanded dimension contract', () => {
  it('localizes every contracted dimension and keeps unknown dimensions readable', () => {
    const dimensions = [
      'language',
      'signage',
      'place_name',
      'road',
      'driving_side',
      'terrain',
      'waterbody',
      'vegetation',
      'building',
      'vehicle',
      'climate',
      'culture',
      'infrastructure',
    ]
    for (const dimension of dimensions) {
      expect(dimensionLabel(dimension)).not.toBe(dimension)
      expect(dimensionLabel(dimension).length).toBeGreaterThan(0)
    }
    expect(dimensionLabel('mystery_dimension')).toBe('mystery_dimension')
  })

  it('orders expanded dimensions by evidence value and keeps unknown dimensions last', () => {
    const groups = groupClues([
      { dimension: 'mystery', value: 'x', confidence: 0.9 },
      { dimension: 'culture', value: 'festival', confidence: 0.5 },
      { dimension: 'place_name', value: 'kyoto', confidence: 0.6 },
      { dimension: 'language', value: 'ja', confidence: 0.6 },
    ])
    expect(groups.map((group) => group.dimension)).toEqual([
      'language',
      'place_name',
      'culture',
      'mystery',
    ])
  })

  it('localizes synced dimension values without losing existing translations', () => {
    expect(clueValueLabel('left')).toBe('左行交通')
    expect(clueValueLabel('right')).toBe('右行交通')
    expect(clueValueLabel('eiffel')).toBe('埃菲尔铁塔')
    expect(clueValueLabel('station')).toBe('车站')
    expect(clueValueLabel('tropical_rainforest')).toBe('热带雨林')
    expect(clueValueLabel('untranslated_value')).toBe('untranslated_value')
  })

  it('keeps user evidence over higher-confidence OCR for the same value', () => {
    const groups = groupClues([
      { dimension: 'road', value: 'left', confidence: 0.95, source: 'ocr' },
      { dimension: 'road', value: 'left', confidence: 0.6, source: 'user', evidence: '用户确认' },
    ])
    expect(groups[0]?.items).toHaveLength(1)
    expect(groups[0]?.items[0]?.source).toBe('user')
    expect(groups[0]?.items[0]?.evidence).toBe('用户确认')
  })

  it('marks reference-only and catalog-validated clues separately from provenance', () => {
    const displayOnly: Clue = {
      dimension: 'road',
      value: 'street',
      confidence: 0.5,
      source: 'semantic',
      extra: { display_only: true, affects_ranking: false },
    }
    expect(isReferenceOnly(displayOnly)).toBe(true)
    expect(isReferenceOnly({ dimension: 'road', value: 'street', confidence: 0.5 })).toBe(false)
    expect(isCatalogValidated({ ...displayOnly, extra: { validated: true } })).toBe(true)
    expect(isCatalogValidated(displayOnly)).toBe(false)
  })

  it('counts cards after grouping and dedupe', () => {
    expect(
      countGroupedClues([
        { dimension: 'signage', value: 'A', confidence: 0.5 },
        { dimension: 'signage', value: 'A', confidence: 0.9 },
        { dimension: 'signage', value: 'B', confidence: 0.7 },
      ]),
    ).toBe(2)
  })

  it('tracks the strongest confidence and user evidence per group', () => {
    const groups = groupClues([
      { dimension: 'road', value: 'left_hand', confidence: 0.3, source: 'user' },
      { dimension: 'road', value: 'highway', confidence: 0.8, source: 'rule' },
    ])
    expect(groups[0]?.maxConfidence).toBe(0.8)
    expect(groups[0]?.hasUserEvidence).toBe(true)
  })
})

describe('clue provenance', () => {
  it('distinguishes user, OCR, local analysis, model and heuristic sources', () => {
    expect(clueProvenance({ dimension: 'language', value: 'ja', confidence: 0.9, source: 'user' })).toBe('user')
    expect(clueProvenance({ dimension: 'signage', value: '出口', confidence: 0.8, source: 'ocr' })).toBe('ocr')
    expect(clueProvenance({ dimension: 'vegetation', value: 'desert', confidence: 0.4, source: 'point_click' })).toBe('observed')
    expect(
      clueProvenance({
        dimension: 'language',
        value: 'ja',
        confidence: 0.7,
        source: 'point_click',
        extra: { origin_source: 'ocr' },
      }),
    ).toBe('ocr')
    expect(clueProvenance({ dimension: 'building', value: 'gothic', confidence: 0.4, source: 'building' })).toBe('heuristic')
    expect(clueProvenance({ dimension: 'other', value: 'x', confidence: 0.2, source: 'streetclip' })).toBe('model')
    expect(clueProvenance({ dimension: 'other', value: 'x', confidence: 0.2 })).toBe('unknown')
  })

  it('localizes provenance labels and treats unknown analyzers as heuristic', () => {
    expect(clueSourceLabel({ dimension: 'language', value: 'ja', confidence: 0.9, source: 'user' })).toBe('你补充的')
    expect(clueSourceLabel({ dimension: 'signage', value: '出口', confidence: 0.8, source: 'ocr' })).toBe('图片文字')
    expect(clueSourceLabel({ dimension: 'vegetation', value: 'desert', confidence: 0.4, source: 'vegetation' })).toBe('推断')
    expect(clueSourceLabel({ dimension: 'other', value: 'x', confidence: 0.2, source: 'thermal' })).toBe('推断')
    expect(clueSourceLabel({ dimension: 'other', value: 'x', confidence: 0.2 })).toBe('')
  })
})

describe('evidence-priority defaults and preview cap', () => {
  it('opens the user-supplemented group first', () => {
    const groups = groupClues([
      { dimension: 'vegetation', value: 'desert', confidence: 0.95 },
      { dimension: 'road', value: 'left_hand', confidence: 0.3, source: 'user' },
    ])
    expect(defaultOpenDimensions(groups)).toEqual(['road'])
  })

  it('falls back to the strongest clue when no user evidence exists', () => {
    const groups = groupClues([
      { dimension: 'vegetation', value: 'desert', confidence: 0.4 },
      { dimension: 'building', value: 'gothic', confidence: 0.8 },
    ])
    expect(defaultOpenDimensions(groups)).toEqual(['building'])
    expect(defaultOpenDimensions([])).toEqual([])
  })

  it('caps long OCR groups and expands to every item', () => {
    const lines = Array.from({ length: 7 }, (_, index) => `OCR-${index}`)
    const capped = cluePreview(lines, 4)
    expect(capped.visible).toHaveLength(4)
    expect(capped.hiddenCount).toBe(3)
    const expanded = cluePreview(lines, lines.length)
    expect(expanded.visible).toHaveLength(7)
    expect(expanded.hiddenCount).toBe(0)
  })

  it('marks low or unknown confidence as low without hiding it', () => {
    expect(isLowConfidence(0.2)).toBe(true)
    expect(isLowConfidence(Number.NaN)).toBe(true)
    expect(isLowConfidence(0.8)).toBe(false)
  })
})
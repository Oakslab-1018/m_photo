import { describe, expect, it } from 'vitest'
import type { Clue } from '@/types'
import { clueRegionBox } from '@/utils/clueRegion'

function clue(extra: Record<string, unknown>): Clue {
  return {
    dimension: 'signage',
    value: '出口',
    confidence: 0.8,
    source: 'ocr',
    extra,
  }
}

describe('clueRegionBox', () => {
  it('returns a real per-clue normalized box', () => {
    expect(clueRegionBox(clue({ bbox_norm: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 } }))).toEqual({
      x: 0.1,
      y: 0.2,
      w: 0.3,
      h: 0.4,
    })
  })

  it('falls back to the analyzed source region when the clue box is absent or invalid', () => {
    const regionBboxNorm = { x: 0.05, y: 0.1, w: 0.5, h: 0.6 }
    expect(clueRegionBox(clue({ region_bbox_norm: regionBboxNorm }))).toEqual(regionBboxNorm)
    expect(
      clueRegionBox(
        clue({
          bbox_norm: { x: 0.7, y: 0.7, w: 0.4, h: 0.3 },
          region_bbox_norm: regionBboxNorm,
        }),
      ),
    ).toEqual(regionBboxNorm)
  })

  it('prefers a valid per-clue box over the source-region fallback', () => {
    const bboxNorm = { x: 0.2, y: 0.3, w: 0.1, h: 0.05 }
    expect(
      clueRegionBox(
        clue({
          bbox_norm: bboxNorm,
          region_bbox_norm: { x: 0, y: 0, w: 1, h: 1 },
        }),
      ),
    ).toEqual(bboxNorm)
  })

  it.each([
    ['missing fields', { x: 0.1, y: 0.1, w: 0.2 }],
    ['string coordinates', { x: '0.1', y: 0.1, w: 0.2, h: 0.2 }],
    ['NaN', { x: Number.NaN, y: 0.1, w: 0.2, h: 0.2 }],
    ['infinite value', { x: 0, y: 0, w: Number.POSITIVE_INFINITY, h: 0.2 }],
    ['negative origin', { x: -0.1, y: 0, w: 0.2, h: 0.2 }],
    ['zero width', { x: 0.1, y: 0.1, w: 0, h: 0.2 }],
    ['zero height', { x: 0.1, y: 0.1, w: 0.2, h: 0 }],
    ['out of horizontal bounds', { x: 0.9, y: 0.1, w: 0.2, h: 0.2 }],
    ['out of vertical bounds', { x: 0.1, y: 0.9, w: 0.2, h: 0.2 }],
  ])('rejects malformed geometry: %s', (_label, bbox) => {
    expect(clueRegionBox(clue({ bbox_norm: bbox }))).toBeNull()
  })

  it('never infers geometry from clue text or values', () => {
    const textClue: Clue = {
      dimension: 'other',
      value: 'x=0.1 y=0.2 w=0.3 h=0.4',
      confidence: 0.9,
      evidence: 'bbox_norm {x: 0.1, y: 0.2, w: 0.3, h: 0.4}',
    }
    expect(clueRegionBox(textClue)).toBeNull()
    expect(clueRegionBox({ ...textClue, extra: { note: '0,0,1,1' } })).toBeNull()
  })

  it('returns null for missing clues and malformed extras', () => {
    expect(clueRegionBox(null)).toBeNull()
    expect(clueRegionBox(undefined)).toBeNull()
    expect(clueRegionBox({ ...clue({}), extra: [] as unknown as Record<string, unknown> })).toBeNull()
  })

  it('accepts the same tiny floating-point allowance as the backend validator', () => {
    expect(clueRegionBox(clue({ bbox_norm: { x: 0.8, y: 0.8, w: 0.2 + 1e-10, h: 0.2 } }))).toEqual({
      x: 0.8,
      y: 0.8,
      w: 0.2 + 1e-10,
      h: 0.2,
    })
  })
})

import { describe, expect, it } from 'vitest'
import type { HypothesisAction } from '@/types'
import { actionOptions, answerPayload, groupActionOptions, isQuestionAction } from '@/utils/dialogue'

describe('dialogue action options', () => {
  it('normalizes options, localizes fallback labels and drops malformed entries', () => {
    const action: HypothesisAction = {
      kind: 'ask',
      payload: {
        dimensions: ['climate', 'language'],
        options: [
          { dimension: 'climate', value: 'arid', label: '干旱气候' },
          { dimension: 'language', value: 'ja' },
          { dimension: 'climate', value: 'arid', label: '重复项' },
          { dimension: '', value: 'x' },
          { dimension: 'road', value: '' },
        ],
      },
    }
    expect(actionOptions(action)).toEqual([
      { dimension: 'climate', value: 'arid', label: '干旱气候' },
      { dimension: 'language', value: 'ja', label: '日语' },
    ])
  })

  it('tolerates missing payloads and unknown dimensions', () => {
    expect(actionOptions(null)).toEqual([])
    expect(groupActionOptions(undefined)).toEqual([])
    expect(
      groupActionOptions({
        kind: 'ask',
        payload: { options: [{ dimension: 'mystery_dim', value: 'x' }] },
      }),
    ).toEqual([
      {
        dimension: 'mystery_dim',
        label: 'mystery_dim',
        options: [{ dimension: 'mystery_dim', value: 'x', label: 'x' }],
      },
    ])
  })

  it('treats ask/request_clue and option-bearing actions as questions', () => {
    expect(isQuestionAction({ kind: 'ask' })).toBe(true)
    expect(isQuestionAction({ kind: 'request_clue' })).toBe(true)
    expect(
      isQuestionAction({
        kind: 'reason',
        payload: { options: [{ dimension: 'climate', value: 'arid', label: '干旱气候' }] },
      }),
    ).toBe(true)
    expect(isQuestionAction({ kind: 'confirm', payload: { iso: 'JP' } })).toBe(false)
    expect(isQuestionAction({ kind: 'done' })).toBe(false)
    expect(isQuestionAction(null)).toBe(false)
  })

  it('groups chips by dimension in payload order and emits answer payloads', () => {
    const groups = groupActionOptions({
      kind: 'ask',
      payload: {
        options: [
          { dimension: 'climate', value: 'arid' },
          { dimension: 'climate', value: 'temperate' },
          { dimension: 'language', value: 'ja' },
        ],
      },
    })
    expect(groups.map((group) => group.dimension)).toEqual(['climate', 'language'])
    expect(groups[0]?.label).toBe('气候特征')
    expect(groups[0]?.options).toHaveLength(2)
    expect(answerPayload(groups[1]!.options[0]!)).toEqual({
      type: 'answer',
      dimension: 'language',
      value: 'ja',
    })
  })
})
import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import CandidatePanel from '@/components/CandidatePanel.vue'
import type {
  ConfidenceRoute,
  CountryScore,
  GuidedDialogue,
  HypothesisAction,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import {
  confidenceCountryCandidates,
  countryConfidenceShortcut,
  selectTopGuidedCandidates,
} from '@/utils/guided'

function target(id: string, label: string): LocationTarget {
  return { level: 'country', id, label, country_iso: id }
}

function candidate(
  id: string,
  label: string,
  score?: number | null,
): LocationCandidate {
  return { target: target(id, label), score, score_type: 'hybrid_rank' }
}

function legacyScore(iso: string, name: string): CountryScore {
  return { iso, name, name_zh: name, score: 0.8, lat: 35, lng: 135 }
}

function renderPanel(overrides: Record<string, unknown> = {}): Promise<string> {
  return renderToString(
    createSSRApp(CandidatePanel, {
      scores: [],
      confirmedIso: null,
      selectedIso: null,
      exclusions: [],
      busy: false,
      countryOptions: [],
      cityBusy: false,
      cityNotice: null,
      ...overrides,
    }),
  )
}

describe('guided candidate Top-5', () => {
  it('dedupes by target, sorts finite scores descending and fills with catalog-only rows', () => {
    const candidates = [
      candidate('AA', 'Alpha', 0.4),
      candidate('BB', 'Beta', 0.9),
      candidate('CC', 'Catalog first', null),
      candidate('DD', 'Delta', 0.7),
      candidate('AA', 'Alpha duplicate', 0.95),
      candidate('EE', 'Echo', 0.2),
      candidate('FF', 'Catalog second', null),
    ]

    const top = selectTopGuidedCandidates(candidates, 'country')
    expect(top.map((item) => item.target.id)).toEqual(['AA', 'BB', 'DD', 'EE', 'CC'])
    expect(top[0]?.score).toBe(0.95)
    expect(top).toHaveLength(5)
  })

  it('renders the ranked Top-5 in the country tab and labels scores as non-probability', async () => {
    const candidates = [
      candidate('AA', 'Alpha', 0.4),
      candidate('BB', 'Beta', 0.9),
      candidate('CC', 'Catalog first', null),
      candidate('DD', 'Delta', 0.7),
      candidate('AA', 'Alpha duplicate', 0.95),
      candidate('EE', 'Echo', 0.2),
      candidate('FF', 'Catalog second', null),
    ]
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: candidates,
    })

    expect(html.indexOf('Alpha')).toBeLessThan(html.indexOf('Beta'))
    expect(html.indexOf('Beta')).toBeLessThan(html.indexOf('Delta'))
    expect(html.indexOf('Delta')).toBeLessThan(html.indexOf('Echo'))
    expect(html.indexOf('Echo')).toBeLessThan(html.indexOf('Catalog first'))
    expect(html).not.toContain('Catalog second')
    expect(html).toContain('>5<')
    expect(html).toContain('候选可信度前五 · 排序分仅供参考，不代表概率')
    expect(html).toContain('排序分，非概率')
    expect(html).toContain('目录候选（无模型评分）')
  })

  it('labels the current guided level and removes the legacy city directory tab', async () => {
    const adminTarget: LocationTarget = {
      level: 'admin1',
      id: 'JP.01',
      label: '北海道',
      country_iso: 'JP',
      admin1_id: '01',
    }
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedStage: 'admin1',
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: [{ target: adminTarget, score: 0.66 }],
      scores: [legacyScore('JP', '日本')],
    })

    expect(html).toContain('省份 / 地区')
    expect(html).toContain('北海道')
    expect(html).toContain('>1<')
    expect(html).not.toContain('查询城市候选')
  })

  it('hides count, names and actions while observations are locked', async () => {
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedCandidatesUnlocked: false,
      guidedConfirmationReady: false,
      guidedCandidates: [candidate('JP', '日本', 0.9)],
      scores: [legacyScore('JP', '日本')],
    })

    expect(html).toContain('先核对观察，再查看地点候选')
    expect(html).not.toContain('日本')
    expect(html).not.toContain('>1<')
    expect(html).not.toContain('确认国家 / 地区')
  })

  it('never falls back to legacy scores while guided state is loading', async () => {
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: false,
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: [candidate('JP', '日本', 0.9)],
      scores: [legacyScore('JP', '日本')],
    })

    expect(html).toContain('正在读取服务器保存的引导状态')
    expect(html).not.toContain('日本')
    expect(html).not.toContain('候选国家推断')
  })

  it('keeps rankings visible but disables location actions for a new pending observation', async () => {
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: false,
      guidedCandidates: [candidate('JP', '日本', 0.9)],
    })

    expect(html).toContain('日本')
    expect(html).toContain('新增观察待核对')
    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })
})

describe('country confidence routing', () => {
  function confidence(
    tier: ConfidenceRoute['tier'],
    action: ConfidenceRoute['action'],
    isos: string[],
  ): ConfidenceRoute {
    return {
      tier,
      action,
      top3: isos.map((iso, index) => ({
        iso,
        name: iso,
        name_zh: iso,
        score: 0.9 - index * 0.01,
        score_type: 'fusion_rank',
      })),
      leader_changed: false,
      reason: '服务器排序说明',
      policy_version: 'test',
    }
  }

  function countryDialogue(overrides: Partial<GuidedDialogue> = {}): GuidedDialogue {
    return {
      version: 1,
      revision: 1,
      stage: 'country',
      confirmed: { country: null, admin1: null, city: null },
      exclusions: [],
      clues: [],
      pending_question: null,
      ...overrides,
    }
  }

  it('requires the matching high action kind and permits routine initial machine clues', () => {
    const route = confidence('high', 'direct_top1', ['JP', 'CN', 'US'])
    const pending = countryDialogue({
      clues: [
        {
          id: 'machine-clue',
          dimension: 'language',
          value: 'ja',
          description: '日语招牌',
          source: 'model',
          confidence: 0.8,
          status: 'pending',
        },
      ],
      pending_question: {
        id: 'review-machine',
        kind: 'review_clue',
        clue_id: 'machine-clue',
        message: '这条观察准确吗？',
      },
    })
    const highAction: HypothesisAction = { kind: 'result', message: '高置信结果' }
    const reviewAction: HypothesisAction = { kind: 'confirm', message: '请确认' }

    expect(countryConfidenceShortcut(pending, highAction, route)).toBe('high')
    expect(countryConfidenceShortcut(pending, reviewAction, route)).toBeNull()
    expect(countryConfidenceShortcut(pending, highAction, confidence('medium', 'confirm_candidates', ['JP', 'CN']))).toBeNull()
  })

  it('enables medium and exhausted-low manual confirmation only when action and country match', () => {
    const action: HypothesisAction = { kind: 'confirm', message: '请确认候选' }
    const medium = confidence('medium', 'confirm_candidates', ['JP', 'CN', 'US'])
    const exhaustedLow = confidence('low', 'confirm_candidates', ['JP', 'CN', 'US'])

    expect(countryConfidenceShortcut(countryDialogue(), action, medium)).toBe('candidates')
    expect(countryConfidenceShortcut(countryDialogue(), action, exhaustedLow)).toBe('candidates')
    expect(countryConfidenceShortcut(
      countryDialogue({ stage: 'admin1', confirmed: { country: target('JP', '日本'), admin1: null, city: null } }),
      action,
      medium,
    )).toBeNull()
    expect(countryConfidenceShortcut(countryDialogue(), action, medium, true)).toBeNull()
  })

  it('keeps correction questions above the confidence shortcut', () => {
    const correction = countryDialogue({
      pending_question: {
        id: 'correction',
        kind: 'ask',
        message: '是否应用这条纠正？',
        options: [{ value: 'apply', label: '应用纠正' }],
      },
    })
    const action: HypothesisAction = { kind: 'result', message: '高置信结果' }

    expect(countryConfidenceShortcut(
      correction,
      action,
      confidence('high', 'direct_top1', ['JP', 'CN']),
    )).toBeNull()
  })

  it('joins top3 rows to existing server targets in route order and preserves target metadata', () => {
    const japan = candidate('JP', '日本', 0.8)
    japan.reason = '国家模型证据'
    const china = candidate('CN', '中国', 0.79)
    const us = candidate('US', '美国', 0.5)
    const route = confidence('medium', 'confirm_candidates', ['CN', 'JP', 'US'])

    const rows = confidenceCountryCandidates(route, [japan, china, us], 3)

    expect(rows.map((item) => item.target.id)).toEqual(['CN', 'JP', 'US'])
    expect(rows[1]?.target).toEqual(japan.target)
    expect(rows[1]?.reason).toBe('国家模型证据')
    expect(rows[1]?.score).toBe(0.89)
    expect(rows[1]?.score_type).toBe('fusion_rank')
  })

  it('omits rows without a server target and never synthesizes a country', () => {
    const route = confidence('high', 'direct_top1', ['ZZ'])
    expect(confidenceCountryCandidates(route, [candidate('JP', '日本', 0.8)], 1)).toEqual([])
  })
})

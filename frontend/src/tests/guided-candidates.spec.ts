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

function cityTarget(id: string, label: string, admin1Id = 'CN.32'): LocationTarget {
  return {
    level: 'city',
    id,
    label,
    country_iso: 'CN',
    city_id: id,
    admin1_id: admin1Id,
    lat: 30.67,
    lng: 104.07,
  }
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

  it('orders real city scores first and keeps unscored proposals after them without inventing scores', () => {
    const models = Array.from({ length: 5 }, (_, index) => ({
      target: cityTarget(String(2000 + index), `模型城市 ${index + 1}`),
      score: 0.9 - index * 0.1,
      score_type: 'city_text_cosine',
    }))
    const chengdu: LocationCandidate = {
      target: cityTarget('1815286', '成都'),
      reason: '自由文本提议',
      selection_basis: 'user_text',
    }
    const searched: LocationCandidate = {
      target: cityTarget('1816670', '重庆', 'CN.33'),
      reason: '全国城市目录检索',
      selection_basis: 'catalog_search',
    }

    const full = selectTopGuidedCandidates([...models, chengdu, searched], 'city')

    expect(full.map((item) => item.target.id)).toEqual([
      '2000',
      '2001',
      '2002',
      '2003',
      '2004',
    ])
    expect(full.every((item) => Number.isFinite(item.score))).toBe(true)

    const withSpace = selectTopGuidedCandidates(
      [...models.slice(0, 2), chengdu, searched],
      'city',
    )

    expect(withSpace.map((item) => item.target.id)).toEqual([
      '2000',
      '2001',
      '1815286',
      '1816670',
    ])
    expect(withSpace.slice(2).map((item) => item.score)).toEqual([undefined, undefined])
    expect(withSpace.slice(2).map((item) => item.selection_basis)).toEqual([
      'user_text',
      'catalog_search',
    ])
  })

  it('labels a mixed user-text proposal without claiming a top-confidence ranking', async () => {
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedStage: 'city',
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: [
        {
          target: cityTarget('1815286', '成都'),
          reason: '自由文本提议',
          selection_basis: 'user_text',
        },
        {
          target: cityTarget('2000', '模型城市'),
          score: 0.9,
          score_type: 'city_text_cosine',
        },
      ],
    })

    expect(html.indexOf('模型城市')).toBeLessThan(html.indexOf('成都'))
    expect(html).toContain('线索匹配候选')
    expect(html).toContain('标有“线索匹配候选”的条目来自你输入的文字。')
    expect(html).toContain('含线索匹配')
    expect(html).not.toContain('候选可信度前五')
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
    expect(html).toContain('候选前五')
    expect(html).toContain('匹配 0.900（不是概率）')
    expect(html).toContain('目录候选')
  })

  it('describes catalog-only rows as directory order without inventing a ranking', async () => {
    const candidates: LocationCandidate[] = Array.from({ length: 5 }, (_, index) => ({
      target: cityTarget(`city-${index + 1}`, `目录城市 ${index + 1}`),
      score: null,
      score_type: null,
    }))
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedStage: 'city',
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: candidates,
    })

    expect(html).toContain('目录候选（5 项）')
    expect(html).toContain(
      '按目录顺序显示；可输入城市名称，或补充照片里的线索。',
    )
    expect(html).toContain('目录候选（5 项）')
    expect(html).not.toContain('候选可信度前五')
    expect(html).not.toContain('排序分仅供参考')
  })

  it('labels the current guided level and removes the legacy city directory tab', async () => {
    const city: LocationTarget = {
      level: 'city',
      id: '1001',
      label: '札幌',
      country_iso: 'JP',
      city_id: '1001',
      admin1_id: 'JP.01',
    }
    const html = await renderPanel({
      guidedMode: true,
      guidedDataReady: true,
      guidedStage: 'city',
      guidedCandidatesUnlocked: true,
      guidedConfirmationReady: true,
      guidedCandidates: [{ target: city, score: 0.66 }],
      scores: [legacyScore('JP', '日本')],
    })

    expect(html).toContain('城市候选')
    expect(html).toContain('札幌')
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

    expect(html).toContain('先核对观察，再查看候选')
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

    expect(html).toContain('正在加载…')
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
    expect(html).toContain('有新的观察待核对，暂时不能确认或排除。')
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
      countryDialogue({ stage: 'city', confirmed: { country: target('JP', '日本'), admin1: null, city: null } }),
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

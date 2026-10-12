import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import GuidedDialoguePanel from '@/components/GuidedDialoguePanel.vue'
import type { GuidedDialogue, LocationCandidate, LocationTarget } from '@/types'
import { safeGuidedReceipt, safeGuidedText } from '@/utils/guided'

const japan: LocationTarget = { level: 'country', id: 'JP', label: '日本', country_iso: 'JP' }
const hokkaido: LocationTarget = {
  level: 'admin1',
  id: 'JP.01',
  label: '北海道',
  country_iso: 'JP',
  admin1_id: '01',
}
const sapporo: LocationTarget = {
  level: 'city',
  id: '1001',
  label: '札幌',
  country_iso: 'JP',
  admin1_id: 'JP.01',
  city_id: '1001',
  lat: 43.06,
  lng: 141.35,
}

function dialogue(overrides: Partial<GuidedDialogue> = {}): GuidedDialogue {
  return {
    version: 1,
    revision: 7,
    stage: 'country',
    confirmed: { country: null, admin1: null, city: null },
    exclusions: [],
    clues: [],
    pending_question: null,
    ...overrides,
  }
}

function render(
  dialogueValue: GuidedDialogue,
  candidates: LocationCandidate[] = [],
  props: Record<string, unknown> = {},
): Promise<string> {
  return renderToString(
    createSSRApp(GuidedDialoguePanel, {
      dialogue: dialogueValue,
      candidates,
      loading: false,
      applying: false,
      externalBusy: false,
      error: null,
      conflict: null,
      canRetry: false,
      receipt: null,
      ...props,
    }),
  )
}

describe('GuidedDialoguePanel rendering', () => {
  it('renders the Chinese three-step path without internal stage names or revisions', async () => {
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: hokkaido, city: null },
      }),
    )

    expect(html).toContain('国家 / 地区')
    expect(html).toContain('省份 / 地区')
    expect(html).toContain('城市')
    expect(html).toContain('日本')
    expect(html).toContain('北海道')
    expect(html).not.toContain('admin1')
    expect(html).not.toContain('revision')
    expect(html).not.toContain('JP.01')
    expect(html).not.toContain('>7<')
  })

  it('shows one pending question with a reason and an explicit unknown-not-rejection action', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'q-1',
          kind: 'ask',
          message: '这里的气候更接近哪一种？',
          options: [
            { value: 'arid', label: '干旱气候' },
            { value: 'temperate', label: '温带气候' },
          ],
        },
      }),
    )

    expect(html).toContain('这里的气候更接近哪一种？')
    expect(html).toContain('有助于区分当前候选')
    expect(html).toContain('干旱气候')
    expect(html).toContain('温带气候')
    expect(html).toContain('不确定 / 无法判断（不代表排除）')
    expect(html).toContain('本轮仅此一问')
  })

  it('renders catalog candidates without inventing scores and keeps representative-point copy', async () => {
    const html = await render(dialogue({ stage: 'city' }), [
      { target: { ...sapporo, level: 'city' }, reason: '目录匹配到城市名' },
    ])

    expect(html).toContain('目录候选（无模型评分）')
    expect(html).toContain('坐标为代表点，非照片 GPS')
    expect(html).toContain('确认城市')
    expect(html).toContain('排除城市')
    expect(html).not.toContain('概率 1')
  })

  it('re-ranks and caps an unordered full candidate list at five in the component', async () => {
    const country = (id: string, label: string, score: number): LocationCandidate => ({
      target: { level: 'country', id, label, country_iso: id },
      score,
    })
    const candidates = [
      country('JP', '低分国家甲', 0.2),
      country('US', '美国', 0.9),
      country('CN', '中国', 0.4),
      country('KR', '韩国', 0.8),
      country('GB', '低分国家乙', 0.1),
      country('FR', '法国', 0.7),
      country('DE', '德国', 0.6),
      country('IT', '低分国家丙', 0.3),
    ]
    const html = await render(dialogue({ stage: 'country' }), candidates)

    expect(html.match(/<article class="guided-candidate-card"/g)).toHaveLength(5)
    expect(html.indexOf('美国')).toBeLessThan(html.indexOf('韩国'))
    expect(html.indexOf('韩国')).toBeLessThan(html.indexOf('法国'))
    expect(html.indexOf('法国')).toBeLessThan(html.indexOf('德国'))
    expect(html.indexOf('德国')).toBeLessThan(html.indexOf('中国'))
    expect(html).not.toContain('低分国家甲')
    expect(html).not.toContain('低分国家乙')
    expect(html).not.toContain('低分国家丙')
  })

  it('labels rejected evidence as evidence-only, never as geographic exclusion', async () => {
    const html = await render(
      dialogue({
        clues: [
          {
            id: 'clue-1',
            dimension: 'language',
            value: 'ja',
            description: '招牌文字呈日语特征',
            source: 'model',
            confidence: 0.8,
            status: 'rejected',
          },
        ],
      }),
    )

    expect(html).toContain('已标记不准确')
    expect(html).toContain('排除证据也不等于地理排除')
    expect(html).toContain('恢复待核对')
  })

  it('shows an explicit resolved city result', async () => {
    const html = await render(
      dialogue({
        stage: 'resolved',
        confirmed: { country: japan, admin1: hokkaido, city: sapporo },
      }),
    )

    expect(html).toContain('最终结果')
    expect(html).toContain('札幌')
    expect(html).toContain('城市坐标仅为代表点')
  })

  it('labels backend correction questions without exposing the raw dimension', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'correction-1',
          kind: 'ask',
          dimension: 'language',
          message: '是否应用这条纠正？',
          options: [
            { value: 'apply', label: '应用纠正' },
            { value: 'keep', label: '保留原证据' },
          ],
        },
      }),
    )

    expect(html).toContain('核对纠正')
    expect(html).toContain('确认后才更新证据或排除记录')
    expect(html).toContain('应用纠正')
    expect(html).toContain('保留原证据')
    expect(html).not.toContain('文字语言')
  })

  it('disables guided actions while an external point/feedback/upload operation is applying', async () => {
    const html = await render(dialogue(), [{ target: japan }], { externalBusy: true })

    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })

  it('keeps free-text catalog search as a non-confirming action', async () => {
    const html = await render(dialogue({ stage: 'admin1' }))

    expect(html).toContain('输入省份 / 地区名称进行目录检索')
    expect(html).toContain('搜索 / 提交文字')
    expect(html).toContain('不会自动确认地点或进入下一级')
  })

  it('builds a natural review question from the localized clue instead of raw enums', async () => {
    const html = await render(
      dialogue({
        clues: [
          {
            id: 'building-generic',
            dimension: 'building',
            value: 'generic',
            description: '建筑外观没有明显地域特征',
            source: 'building',
            confidence: 0.6,
            status: 'pending',
          },
        ],
        pending_question: {
          id: 'review-building-generic',
          kind: 'review_clue',
          clue_id: 'building-generic',
          dimension: 'building',
          message: '请复核线索：建筑风格=generic。这条线索是否正确？',
        },
      }),
    )

    expect(html).toContain('我们初步判断这是‘普通 / 不明显’的建筑风格，您觉得准确吗？')
    expect(html).toContain('待核对')
    expect(html).toContain('普通 / 不明显')
    expect(html).not.toContain('building=generic')
    expect(html).not.toContain('通用/不明显')
  })

  it('renders every clue without the old six-item cap and lifts new clues first', async () => {
    const clues = Array.from({ length: 9 }, (_, index) => ({
      id: `clue-${index}`,
      dimension: 'other',
      value: `V${index}`,
      description: `观察描述 ${index}`,
      source: 'model',
      confidence: 0.5,
      status: 'pending' as const,
    }))
    const html = await render(dialogue({ clues }), [], { recentClueIds: ['clue-8'] })

    expect(html).toContain('V0')
    expect(html).toContain('V8')
    expect(html).toContain('新线索')
    expect(html).not.toContain('展开其余')
    expect(html.indexOf('V8')).toBeLessThan(html.indexOf('V0'))
    expect(html.indexOf('观察描述 8')).toBeLessThan(html.indexOf('观察描述 0'))
  })

  it('hides non-review questions, options, examples and candidates while rankings are locked', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'ask-location',
          kind: 'ask',
          dimension: 'place_name',
          message: '是否确认日本？',
          options: [{ value: 'JP', label: '日本' }],
        },
      }),
      [{ target: japan, score: 0.9 }],
      { candidatesUnlocked: false, locationConfirmationReady: false },
    )

    expect(html).toContain('先核对观察，再查看地点候选')
    expect(html).toContain('请先完成上方观察核对')
    expect(html).not.toContain('日本')
    expect(html).not.toContain('是否确认')
    expect(html).not.toContain('option-chip')
    expect(html).not.toMatch(/class="mini-btn confirm"/)
  })

  it('disables location actions when a newly appended observation needs review', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }], {
      candidatesUnlocked: true,
      locationConfirmationReady: false,
    })

    expect(html).toContain('新增观察待核对，候选排名仍可查看，但暂不能确认或排除地点。')
    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })

  it('offers an explicit stage entry only after every pending clue is handled', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }], {
      candidatesUnlocked: true,
      locationConfirmationReady: true,
    })

    expect(html).toContain('观察核对已完成')
    expect(html).toContain('进入国家 / 地区筛选')
    expect(html).toContain('系统不会自动确认地点')
    expect(html).toContain('guided-next-step-action')
  })

  it('does not offer the next-stage entry while any observation remains pending', async () => {
    const html = await render(
      dialogue({
        clues: [
          {
            id: 'pending-clue',
            dimension: 'language',
            value: 'zh',
            description: '招牌文字仍待核对',
            source: 'model',
            confidence: 0.8,
            status: 'pending',
          },
        ],
      }),
      [{ target: japan, score: 0.9 }],
      { candidatesUnlocked: true, locationConfirmationReady: false },
    )

    expect(html).not.toContain('观察核对已完成')
    expect(html).not.toContain('guided-next-step-action')
    expect(html).toContain('新增观察待核对')
  })

  it('uses the current stage level in the next-stage entry label', async () => {
    const reviewedClue = {
      id: 'reviewed-clue',
      dimension: 'language',
      value: 'ja',
      description: '招牌文字呈日语特征',
      source: 'model',
      confidence: 0.8,
      status: 'confirmed' as const,
    }
    const adminHtml = await render(
      dialogue({
        stage: 'admin1',
        confirmed: { country: japan, admin1: null, city: null },
        clues: [reviewedClue],
      }),
      [{ target: hokkaido }],
      { candidatesUnlocked: true },
    )
    const cityHtml = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: hokkaido, city: null },
        clues: [reviewedClue],
      }),
      [{ target: sapporo }],
      { candidatesUnlocked: true },
    )

    expect(adminHtml).toContain('进入省份 / 地区筛选')
    expect(adminHtml).not.toContain('进入国家 / 地区筛选')
    expect(cityHtml).toContain('进入城市筛选')
    expect(cityHtml).not.toContain('进入省份 / 地区筛选')
  })

  it('asks for additional observable clues when review finishes without candidates', async () => {
    const html = await render(dialogue(), [], { candidatesUnlocked: true })

    expect(html).toContain('观察核对已完成')
    expect(html).toContain('当前层级暂无候选')
    expect(html).toContain('补充照片中可见的招牌、地名或文字')
    expect(html).not.toContain('guided-next-step-action')
  })
})
describe('guided audit text', () => {
  it('does not expose backend stage identifiers or rewrite location labels', () => {
    expect(safeGuidedReceipt('已回退到admin1阶段。')).toContain('省份 / 地区')
    expect(safeGuidedText('已确认城市：Mexico City。')).toBe('已确认城市：Mexico City。')
    expect(safeGuidedText('已回退到city阶段。')).toContain('城市')
    expect(safeGuidedText('请复核线索：building=generic。')).toContain('普通 / 不明显')
    expect(safeGuidedText('请复核线索：building=generic。')).not.toContain('generic')
  })
})


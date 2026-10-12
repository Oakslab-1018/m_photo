import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import GuidedDialoguePanel from '@/components/GuidedDialoguePanel.vue'
import type { GuidedDialogue, LocationCandidate, LocationTarget, ReviewedClue } from '@/types'
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

function clue(id: string, value: string, status: ReviewedClue['status'] = 'pending'): ReviewedClue {
  return {
    id,
    dimension: 'language',
    value,
    description: `观察描述 ${id}`,
    source: 'model',
    confidence: 0.8,
    status,
  }
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

describe('GuidedDialoguePanel review box', () => {
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

  it('renders exactly one server-pinned observation with position and no clue list', async () => {
    const france: LocationTarget = { level: 'country', id: 'FR', label: '法国', country_iso: 'FR' }
    const clues = [
      clue('clue-1', 'zh', 'confirmed'),
      clue('clue-2', 'ja'),
      clue('clue-3', 'ko'),
      clue('clue-4', 'en'),
    ]
    const html = await render(
      dialogue({
        clues,
        pending_question: {
          id: 'review-clue-2',
          kind: 'review_clue',
          clue_id: 'clue-2',
          dimension: 'language',
          message: '请复核线索：language=ja。',
        },
      }),
      [{ target: france, score: 0.9 }],
    )

    expect(html).toContain('第 2 / 4')
    expect(html).toContain('我们初步判断画面中的文字语言是‘日语’，您觉得准确吗？')
    expect(html).toContain('判断准确')
    expect(html).toContain('不准确')
    expect(html).toContain('不确定 / 无法判断（不代表排除）')
    // Only the pinned observation renders: the other clues stay in the left list.
    expect(html).not.toContain('韩语')
    expect(html).not.toContain('英语')
    expect(html).not.toContain('观察描述 clue-3')
    // Candidates never render next to the review box.
    expect(html).not.toContain('法国')
    expect(html).not.toContain('排序分')
    expect(html).not.toContain('guided-candidate-card')
    expect(html).not.toContain('guided-next-step-action')
  })

  it('marks a newly appended pinned observation without re-listing the rest', async () => {
    const clues = [clue('clue-1', 'zh', 'confirmed'), clue('clue-2', 'ja')]
    const html = await render(
      dialogue({
        clues,
        pending_question: {
          id: 'review-clue-2',
          kind: 'review_clue',
          clue_id: 'clue-2',
          message: '请复核线索：language=ja。',
        },
      }),
      [],
      { recentClueIds: ['clue-2'] },
    )

    expect(html).toContain('新线索')
    expect(html).toContain('第 2 / 2')
    expect(html).not.toContain('展开其余')
    expect(html).not.toContain('guided-clue-list')
  })

  it('keeps blocking ask questions answerable while candidates stay hidden', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'ask-location',
          kind: 'ask',
          dimension: 'place_name',
          message: '这里的招牌更接近哪一种？',
          options: [
            { value: 'arid', label: '干旱气候' },
            { value: 'temperate', label: '温带气候' },
          ],
        },
      }),
      [{ target: japan, score: 0.9 }],
      { candidatesUnlocked: false, locationConfirmationReady: false },
    )

    expect(html).toContain('这里的招牌更接近哪一种？')
    expect(html).toContain('干旱气候')
    expect(html).toContain('温带气候')
    expect(html).toContain('option-chip')
    expect(html).toContain('本轮仅此一问')
    expect(html).toContain('不确定 / 跳过此问题')
    expect(html).not.toContain('guided-candidate-card')
    expect(html).not.toContain('候选可信度前五')
  })

  it('keeps the unknown skip available for correction questions', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'correction-unknown',
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

    expect(html).toContain('不确定 / 跳过此问题')
    expect(html).not.toContain('guided-next-step-action')
  })

  it('renders a correction question with its options in the same review box', async () => {
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

  it('keeps a confirm_location prompt generic until the candidate phase', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'confirm-country-jp',
          kind: 'confirm_location',
          message: '是否确认日本？',
          target: japan,
        },
      }),
      [{ target: japan, score: 0.9 }],
    )

    expect(html).toContain('下一步将展示地点候选')
    expect(html).not.toContain('是否确认日本')
    expect(html).not.toContain('guided-candidate-card')
  })

  it('offers the explicit next step only after every observation is handled', async () => {
    const france: LocationTarget = { level: 'country', id: 'FR', label: '法国', country_iso: 'FR' }
    const html = await render(dialogue(), [{ target: france, score: 0.9 }], {
      candidatesUnlocked: true,
      locationConfirmationReady: false,
    })

    expect(html).toContain('观察核对已完成')
    expect(html).toContain('下一步查看地点候选；选择符合照片的地点后再确认。')
    expect(html).toContain('下一步：查看国家 / 地区候选')
    expect(html).toContain('guided-next-step-action')
    // The candidates are already unlocked server-side but stay hidden until the click.
    expect(html).not.toContain('guided-candidate-card')
    expect(html).not.toContain('法国')
    expect(html).not.toContain('排序分')
  })

  it('does not offer the next step while an observation is still pending', async () => {
    const html = await render(
      dialogue({
        clues: [clue('pending-clue', 'zh')],
        pending_question: {
          id: 'review-pending',
          kind: 'review_clue',
          clue_id: 'pending-clue',
          message: '请复核线索：language=zh。',
        },
      }),
      [{ target: japan, score: 0.9 }],
      { candidatesUnlocked: true, locationConfirmationReady: false },
    )

    expect(html).not.toContain('观察核对已完成')
    expect(html).not.toContain('guided-next-step-action')
    expect(html).toContain('第 1 / 1')
  })

  it('does not offer the next step while a blocking ask is unanswered', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'ask-still-open',
          kind: 'ask',
          message: '这段文字是什么？',
          options: [{ value: 'zh', label: '中文' }],
        },
      }),
      [{ target: japan, score: 0.9 }],
      { candidatesUnlocked: true },
    )

    expect(html).not.toContain('guided-next-step-action')
    expect(html).toContain('这段文字是什么？')
  })

  it('uses the current stage level in the next-step label', async () => {
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

    expect(adminHtml).toContain('下一步：查看省份 / 地区候选')
    expect(adminHtml).not.toContain('下一步：查看国家 / 地区候选')
    expect(adminHtml).not.toContain('北海道')
    expect(cityHtml).toContain('下一步：查看城市候选')
    expect(cityHtml).not.toContain('下一步：查看省份 / 地区候选')
    expect(cityHtml).not.toContain('札幌')
  })

  it('renders a prominent backend back button for admin1/city/resolved stages', async () => {
    const adminHtml = await render(
      dialogue({ stage: 'admin1', confirmed: { country: japan, admin1: null, city: null } }),
      [],
      { phase: 'candidates' },
    )
    const resolvedHtml = await render(
      dialogue({
        stage: 'resolved',
        confirmed: { country: japan, admin1: hokkaido, city: sapporo },
      }),
    )

    expect(adminHtml).toContain('返回上一步：国家 / 地区')
    expect(resolvedHtml).toContain('返回上一步：城市')
    expect(resolvedHtml).toContain('最终结果')
  })
})

describe('GuidedDialoguePanel candidate phase', () => {
  it('renders catalog candidates without inventing scores and keeps representative-point copy', async () => {
    const html = await render(
      dialogue({ stage: 'city' }),
      [{ target: { ...sapporo, level: 'city' }, reason: '目录匹配到城市名' }],
      { phase: 'candidates' },
    )

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
    const html = await render(dialogue({ stage: 'country' }), candidates, { phase: 'candidates' })

    expect(html.match(/<article class="guided-candidate-card"/g)).toHaveLength(5)
    expect(html.indexOf('美国')).toBeLessThan(html.indexOf('韩国'))
    expect(html.indexOf('韩国')).toBeLessThan(html.indexOf('法国'))
    expect(html.indexOf('法国')).toBeLessThan(html.indexOf('德国'))
    expect(html.indexOf('德国')).toBeLessThan(html.indexOf('中国'))
    expect(html).not.toContain('低分国家甲')
    expect(html).not.toContain('低分国家乙')
    expect(html).not.toContain('低分国家丙')
  })

  it('offers 返回核对 and 重审上一条观察 in the candidate phase', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }], {
      phase: 'candidates',
      canReviewBack: true,
    })

    expect(html).toContain('返回核对')
    expect(html).toContain('重审上一条观察')
    expect(html).toContain('日本')
    expect(html).not.toContain('观察核对已完成')
  })

  it('keeps the review-back affordance visible and disabled before any review', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }])

    expect(html).toContain('返回上一条观察（重新核对）')
    expect(html).toMatch(/class="ghost-btn"[^>]*disabled[^>]*>\s*返回上一条观察/)
    expect(html).not.toContain('guided-candidate-card')
  })

  it('enables the review-back affordance while reviewing when history exists', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }], {
      canReviewBack: true,
    })

    expect(html).toContain('返回上一条观察（重新核对）')
    expect(html).not.toContain('guided-candidate-card')
  })

  it('disables candidate actions while an external operation is applying', async () => {
    const html = await render(dialogue(), [{ target: japan }], {
      phase: 'candidates',
      externalBusy: true,
    })

    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })

  it('disables location actions when a newly appended observation needs review', async () => {
    const html = await render(dialogue(), [{ target: japan, score: 0.9 }], {
      phase: 'candidates',
      candidatesUnlocked: true,
      locationConfirmationReady: false,
    })

    expect(html).toContain('新增观察待核对，候选排名仍可查看，但暂不能确认或排除地点。')
    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })

  it('shows the candidate empty hint in the candidate phase without a fabricated score', async () => {
    const html = await render(dialogue(), [], { phase: 'candidates' })

    expect(html).toContain('当前层级暂无候选')
    expect(html).toContain('补充照片中可见的招牌、地名或文字')
    expect(html).not.toContain('guided-candidate-card')
  })
})

describe('GuidedDialoguePanel utilities', () => {
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
    expect(html).toContain('普通 / 不明显')
    expect(html).toContain('第 1 / 1')
    expect(html).not.toContain('building=generic')
    expect(html).not.toContain('通用/不明显')
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
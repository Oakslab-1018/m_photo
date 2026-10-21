import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import GuidedDialoguePanel from '@/components/GuidedDialoguePanel.vue'
import type {
  ConfidenceRoute,
  GuidedDialogue,
  LocationCandidate,
  LocationTarget,
  ReviewedClue,
} from '@/types'
import { guidedFreeTextReceiptOutcome, safeGuidedReceipt, safeGuidedText } from '@/utils/guided'

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
      score_type: 'country_rank',
    })),
    leader_changed: false,
    reason: '不应在界面展示的内部排序说明',
    policy_version: 'test',
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
  it('renders the two-step path and ignores deprecated admin1 metadata', async () => {
    const countryHtml = await render(dialogue())
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: hokkaido, city: null },
      }),
    )

    expect(countryHtml).toContain('1 / 2 · 当前步骤')
    expect(html).toContain('国家 / 地区')
    expect(html).toContain('城市')
    expect(html).toContain('日本')
    expect(html).toContain('2 / 2 · 当前步骤')
    expect(html).not.toContain('省份')
    expect(html).not.toContain('北海道')
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

  it('uses country and city in the next-step labels', async () => {
    const reviewedClue = {
      id: 'reviewed-clue',
      dimension: 'language',
      value: 'ja',
      description: '招牌文字呈日语特征',
      source: 'model',
      confidence: 0.8,
      status: 'confirmed' as const,
    }
    const countryHtml = await render(
      dialogue({
        stage: 'country',
        clues: [reviewedClue],
      }),
      [{ target: japan }],
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

    expect(countryHtml).toContain('下一步：查看国家 / 地区候选')
    expect(countryHtml).not.toContain('下一步：查看城市候选')
    expect(countryHtml).not.toContain('日本')
    expect(cityHtml).toContain('下一步：查看城市候选')
    expect(cityHtml).not.toContain('省份')
    expect(cityHtml).not.toContain('北海道')
    expect(cityHtml).not.toContain('札幌')
  })

  it('backtracks directly from city to country and resolved to city', async () => {
    const cityHtml = await render(
      dialogue({ stage: 'city', confirmed: { country: japan, admin1: hokkaido, city: null } }),
      [],
      { phase: 'candidates' },
    )
    const resolvedHtml = await render(
      dialogue({
        stage: 'resolved',
        confirmed: { country: japan, admin1: hokkaido, city: sapporo },
      }),
    )

    expect(cityHtml).toContain('返回上一步：国家 / 地区')
    expect(cityHtml).not.toContain('返回上一步：城市')
    expect(resolvedHtml).toContain('返回上一步：城市')
    expect(resolvedHtml).toContain('最终结果')
    expect(resolvedHtml).toContain('2 / 2 · 已完成')
    expect(resolvedHtml).not.toContain('省份')
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

    expect(html).toContain('新增观察待核对，排名仍可查看，但暂不能确认或排除地点。')
    expect(html).toMatch(/class="mini-btn confirm"[^>]*disabled/)
    expect(html).toMatch(/class="mini-btn reject"[^>]*disabled/)
  })

  it('shows the candidate empty hint in the candidate phase without a fabricated score', async () => {
    const html = await render(dialogue(), [], { phase: 'candidates' })

    expect(html).toContain('当前层级暂无候选')
    expect(html).toContain('补充照片中可见的招牌、地名或文字')
    expect(html).not.toContain('guided-candidate-card')
  })

  it('labels catalog-only city candidates by directory order instead of confidence rank', async () => {
    const candidates: LocationCandidate[] = Array.from({ length: 5 }, (_, index) => ({
      target: {
        level: 'city',
        id: `${1000 + index}`,
        label: `目录城市 ${index + 1}`,
        country_iso: 'JP',
        city_id: `${1000 + index}`,
      },
      score: null,
    }))
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: null, city: null },
      }),
      candidates,
      { phase: 'candidates' },
    )

    expect(html).toContain('目录候选（当前显示 5 项）')
    expect(html).toContain(
      '按目录顺序展示，可输入城市名称查找更多；尚无图像评分，不代表最可能的五个',
    )
    expect(html).toContain('目录候选（无模型评分）')
    expect(html).not.toContain('候选可信度前五')
    expect(html).not.toContain('排序分仅供参考')
  })

  it('keeps a user-text city proposal visible ahead of scored model rows', async () => {
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: hokkaido, city: null },
      }),
      [
        {
          target: {
            level: 'city',
            id: '1815286',
            label: '成都',
            country_iso: 'CN',
            admin1_id: 'CN.32',
            city_id: '1815286',
          },
          reason: '自由文本提议',
          selection_basis: 'user_text',
        },
        ...Array.from({ length: 5 }, (_, index) => ({
          target: {
            level: 'city' as const,
            id: `${2000 + index}`,
            label: `模型城市 ${index + 1}`,
            country_iso: 'JP',
            city_id: `${2000 + index}`,
          },
          score: 0.9 - index * 0.1,
          score_type: 'city_text_cosine',
        })),
      ],
      { phase: 'candidates' },
    )

    expect(html.indexOf('成都')).toBeLessThan(html.indexOf('模型城市 1'))
    expect(html).toContain('线索匹配候选')
    expect(html).toContain('来自你提供的文字线索（非模型概率）')
    expect(html).toContain('线索匹配优先 · 其余按排序分')
    expect(html).toContain('城市文本余弦排序 0.900（排序分，非概率）')
    expect(html).not.toContain('候选可信度前五')
  })
})

describe('GuidedDialoguePanel utilities', () => {
  it('keeps guided map labels on the public country and city path', () => {
    const source = readFileSync(new URL('../components/GeoMap.vue', import.meta.url), 'utf8')

    expect(source).toContain('国家与城市引导候选代表点交互地图')
    expect(source).not.toContain('省份代表点')
    expect(source).not.toContain("case 'admin1'")
  })

  it('renders one conversation composer outside the scroll panel with clue and catalog-name support', async () => {
    const html = await render(dialogue({ stage: 'city' }))

    expect(html).toContain('补充线索，与定位助手对话')
    expect(html).toContain('例如：路牌上是日文，车辆靠左行驶；也可直接输入地名')
    expect(html).toContain('发送线索')
    expect(html).toContain('可直接输入城市名称检索目录')
    expect(html).toContain('提交不会自动确认地点')
    expect(html).toContain('id="guided-conversation-input"')
    expect(html.match(/id="guided-conversation-input"/g)).toHaveLength(1)
    expect(html).not.toContain('搜索 / 提交文字')
    expect(html).toMatch(
      /<div[^>]*class="guided-panel"[^>]*>[\s\S]*<\/div>\s*<form[^>]*class="guided-composer"/,
    )
  })

  it('keeps the composer outside the scroll region before long candidate content', async () => {
    const candidates: LocationCandidate[] = Array.from({ length: 5 }, (_, index) => ({
      target: {
        level: 'city',
        id: `${1000 + index}`,
        label: `长候选城市 ${index + 1}`,
        country_iso: 'JP',
        city_id: `${1000 + index}`,
      },
      score: null,
      reason: '很长的候选说明'.repeat(40),
    }))
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: null, city: null },
      }),
      candidates,
      { phase: 'candidates' },
    )
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8')

    expect(html).toMatch(
      /<section class="guided-dialogue-shell">\s*<div[^>]*class="guided-panel"[^>]*>[\s\S]*<\/div>\s*<form[^>]*class="guided-composer"/,
    )
    expect(css).toMatch(
      /\.guided-dialogue-shell\s*\{[^}]*overflow:\s*hidden;[^}]*display:\s*flex;[^}]*flex-direction:\s*column;/s,
    )
    expect(css).toMatch(/\.guided-panel\s*\{[^}]*overflow-y:\s*auto;/s)
    expect(css).toMatch(/\.guided-composer\s*\{[^}]*flex:\s*0 0 auto;/s)

    const longReview = await render(
      dialogue({
        clues: [clue('long-review', 'ja', 'pending')],
        pending_question: {
          id: 'long-review-question',
          kind: 'review_clue',
          clue_id: 'long-review',
          message: `请核对这条观察：${'很长的观察说明'.repeat(80)}`,
        },
      }),
    )
    expect(longReview).toMatch(
      /<div[^>]*class="guided-panel"[^>]*>[\s\S]*<\/div>\s*<form[^>]*class="guided-composer"/,
    )
  })

  it('shows the current assistant explanation near the lower-stage composer', async () => {
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: null, city: null },
      }),
      [],
      {
        action: {
          kind: 'ask',
          message: '请结合照片中的道路方向和地名，确认城市候选。',
        },
      },
    )

    expect(html).toContain('guided-composer-assistant')
    expect(html).toContain('请结合照片中的道路方向和地名，确认城市候选。')
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
    expect(html).toContain('补充线索，与定位助手对话')
    expect(html).toContain('新线索会交回助手重新核对')
  })

  it('shows the last submitted user context and readable server feedback', async () => {
    const applied = await render(dialogue(), [], {
      freeTextSubmission: {
        text: '路牌上是日文，车辆靠左行驶',
        status: 'applied',
      },
      receipt: {
        event_id: 'event-1',
        status: 'applied',
        summary: '已识别到新的文字线索',
      },
    })
    const failed = await render(dialogue(), [], {
      freeTextSubmission: {
        text: '路牌上是日文，车辆靠左行驶',
        status: 'failed',
        error: '服务端状态已更新',
      },
    })
    const unchanged = await render(dialogue(), [], {
      freeTextSubmission: {
        text: '玉林路',
        status: 'unchanged',
      },
      receipt: {
        event_id: 'event-2',
        status: 'unchanged',
        summary: '该线索此前已保存。',
      },
    })

    expect(applied).toContain('你说')
    expect(applied).toContain('路牌上是日文，车辆靠左行驶')
    expect(applied).toContain('助手反馈：已识别到新的文字线索')
    expect(failed).toContain('服务端状态已更新')
    expect(failed).toContain('草稿已保留')
    expect(unchanged).toContain('玉林路')
    expect(unchanged).toContain('这条线索已记录，无需重复添加。')
    expect(unchanged).toContain('该线索此前已保存')
    expect(unchanged).not.toContain('提交失败')
  })
})

describe('guided free-text receipt mapping', () => {
  it('distinguishes saved-unchanged from unrecognized failure without faking applied', () => {
    expect(guidedFreeTextReceiptOutcome({
      event_id: 'saved',
      status: 'unchanged',
      summary: '该线索此前已保存。',
    })).toEqual({ status: 'unchanged' })
    expect(guidedFreeTextReceiptOutcome({
      event_id: 'new',
      status: 'applied',
      summary: '已记录。',
    })).toEqual({ status: 'applied' })
    expect(guidedFreeTextReceiptOutcome({
      event_id: 'unknown',
      status: 'unrecognized',
      summary: '没有识别到新的线索。',
    })).toEqual({ status: 'failed', error: '没有识别到新的线索。' })
    expect(guidedFreeTextReceiptOutcome({
      event_id: 'mystery',
      status: 'something_else',
      summary: '未定义状态。',
    })).toEqual({
      status: 'unconfirmed',
      error: '服务器未返回可识别的回执，请核对最新状态后重试。',
    })
    expect(guidedFreeTextReceiptOutcome(null)).toEqual({
      status: 'unconfirmed',
      error: '服务器未返回可识别的回执，请核对最新状态后重试。',
    })
  })
})

describe('guided audit text', () => {
  it('does not expose backend stage identifiers or rewrite location labels', () => {
    expect(safeGuidedReceipt('已回退到admin1阶段。')).toContain('城市')
    expect(safeGuidedReceipt('已回退到admin1阶段。')).not.toContain('admin1')
    expect(safeGuidedReceipt('已回退到admin1阶段。')).not.toContain('省份')
    expect(safeGuidedText('已确认城市：Mexico City。')).toBe('已确认城市：Mexico City。')
    expect(safeGuidedText('已回退到city阶段。')).toContain('城市')
    expect(safeGuidedText('请复核线索：building=generic。')).toContain('普通 / 不明显')
    expect(safeGuidedText('请复核线索：building=generic。')).not.toContain('generic')
  })
})

describe('GuidedDialoguePanel country confidence tiers', () => {
  it('renders high top1 with the authoritative explanation and explicit confirmation only', async () => {
    const html = await render(
      dialogue({
        clues: [clue('routine', 'ja')],
        pending_question: {
          id: 'routine-review',
          kind: 'review_clue',
          clue_id: 'routine',
          message: '这条常规观察准确吗？',
        },
      }),
      [],
      {
        action: { kind: 'result', message: '服务器说明：首选日本，请结合照片确认。' },
        confidence: confidence('high', 'direct_top1', ['JP', 'CN', 'US']),
        confidenceShortcut: 'high',
        confidenceCandidates: [
          { target: japan, score: 0.9, reason: '国家模型排序第一' },
          { target: { level: 'country', id: 'CN', label: '中国', country_iso: 'CN' }, score: 0.89 },
          { target: { level: 'country', id: 'US', label: '美国', country_iso: 'US' }, score: 0.7 },
        ],
        locationConfirmationReady: true,
      },
    )

    expect(html).toContain('推荐国家 / 地区')
    expect(html).toContain('服务器说明：首选日本，请结合照片确认。')
    expect(html).toContain('可继续深化城市')
    expect(html).toContain('日本')
    expect(html).toContain('国家模型排序第一')
    expect(html).toContain('补充线索细化')
    expect(html).toContain('确认国家 / 地区')
    expect(html).toContain('这是当前国家预测；如需继续深化城市，可确认该国家后进入下一步。')
    expect(html).not.toContain('仍需要您明确确认')
    expect(html).not.toContain('这条常规观察准确吗')
    expect(html).not.toContain('下一步：查看国家 / 地区候选')
    expect(html).not.toContain('中国')
    expect(html).not.toContain('美国')
    expect(html).not.toContain('不应在界面展示的内部排序说明')
    expect(html).not.toContain('model-only')
    expect(html).not.toContain('0.02')
    expect(html.match(/guided-confidence-card/g)).toHaveLength(1)
  })

  it('renders up to three medium candidates with the server explanation and explicit actions', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'routine-review',
          kind: 'review_clue',
          clue_id: 'routine',
          message: '这条常规观察准确吗？',
        },
      }),
      [],
      {
        action: { kind: 'confirm', message: '服务器说明：日本、中国、美国较接近。' },
        confidence: confidence('medium', 'confirm_candidates', ['JP', 'CN', 'US', 'FR']),
        confidenceShortcut: 'candidates',
        confidenceCandidates: [
          { target: japan, score: 0.8 },
          { target: { level: 'country', id: 'CN', label: '中国', country_iso: 'CN' }, score: 0.79 },
          { target: { level: 'country', id: 'US', label: '美国', country_iso: 'US' }, score: 0.7 },
          { target: { level: 'country', id: 'FR', label: '法国', country_iso: 'FR' }, score: 0.6 },
        ],
        locationConfirmationReady: true,
      },
    )

    expect(html).toContain('国家 / 地区候选待确认')
    expect(html).toContain('服务器说明：日本、中国、美国较接近。')
    expect(html).toContain('日本')
    expect(html).toContain('中国')
    expect(html).toContain('美国')
    expect(html).not.toContain('法国')
    expect(html).not.toContain('这条常规观察准确吗')
    expect(html.match(/guided-confidence-card/g)).toHaveLength(3)
    expect(html.match(/class="mini-btn confirm"/g)).toHaveLength(3)
    expect(html).toContain('请从以上候选中明确确认国家；确认后会进入下一级，未确认前不会推进。')
  })

  it('keeps low confidence on the single question and free-text path', async () => {
    const html = await render(
      dialogue({
        pending_question: {
          id: 'low-question',
          kind: 'ask',
          dimension: 'signage',
          message: '画面中是否有清晰的招牌文字？',
          options: [
            { value: 'clear_sign', label: '有清晰招牌文字' },
            { value: 'no_sign', label: '没有清晰文字' },
          ],
        },
      }),
      [],
      {
        action: {
          kind: 'ask',
          message: '我注意到画面里可能有文字，请先确认能否看清招牌内容。',
          payload: {
            question_id: 'low-question',
            question_kind: 'ask',
            dimension: 'signage',
          },
        },
        confidence: confidence('low', 'ask_clue', ['JP', 'CN', 'US']),
        confidenceShortcut: null,
        confidenceCandidates: [],
      },
    )

    expect(html).toContain('当前信息不足，先补充可观察线索')
    expect(html).toContain('问题说明：')
    expect(html).not.toContain('LLM 解释当前问题')
    expect(html).toContain('我注意到画面里可能有文字，请先确认能否看清招牌内容。')
    expect(html).toContain('画面中是否有清晰的招牌文字？')
    expect(html).toContain('有清晰招牌文字')
    expect(html).toContain('没有清晰文字')
    expect(html).toContain('提交后会重新比较国家候选')
    expect(html).toContain('补充线索，与定位助手对话')
    expect(html).toContain('发送线索')
    expect(html).toContain('type="text"')
    expect(html).not.toContain('guided-confidence-card')
    expect(html).not.toContain('class="mini-btn confirm"')
  })

  it('falls back to the deterministic low question when action metadata does not match', async () => {
    const pending = {
      id: 'low-question',
      kind: 'ask' as const,
      dimension: 'signage',
      message: '画面中是否有清晰的招牌文字？',
    }
    const staleId = await render(dialogue({ pending_question: pending }), [{ target: japan, score: 0.9 }], {
      action: {
        kind: 'ask',
        message: '旧候选包括日本和中国，这条旧说明不应显示。',
        payload: {
          question_id: 'stale-question',
          question_kind: 'ask',
          dimension: 'signage',
        },
      },
      confidence: confidence('low', 'ask_clue', ['JP', 'CN', 'US']),
    })
    const mismatchedKind = await render(dialogue({ pending_question: pending }), [], {
      action: {
        kind: 'ask',
        message: '这条类型不匹配的 LLM 说明不应显示。',
        payload: {
          question_id: 'low-question',
          question_kind: 'request_clue',
          dimension: 'signage',
        },
      },
      confidence: confidence('low', 'ask_clue', ['JP', 'CN', 'US']),
    })
    const nonLow = await render(dialogue({ pending_question: pending }), [], {
      action: {
        kind: 'ask',
        message: '高置信路由不应显示低置信问题说明。',
        payload: {
          question_id: 'low-question',
          question_kind: 'ask',
          dimension: 'signage',
        },
      },
      confidence: confidence('medium', 'confirm_candidates', ['JP', 'CN', 'US']),
    })

    expect(staleId).not.toContain('旧候选包括日本和中国，这条旧说明不应显示。')
    expect(staleId).not.toContain('日本')
    expect(staleId).toContain('画面中是否有清晰的招牌文字？')
    expect(mismatchedKind).not.toContain('这条类型不匹配的 LLM 说明不应显示。')
    expect(mismatchedKind).toContain('画面中是否有清晰的招牌文字？')
    expect(nonLow).not.toContain('高置信路由不应显示低置信问题说明。')
    expect(nonLow).not.toContain('问题说明：')
  })

  it('offers backend-authorized exhausted-low top3 as optional manual confirmation', async () => {
    const html = await render(
      dialogue(),
      [],
      {
        action: { kind: 'confirm', message: '线索不足时的候选，请仅在能够确认时选择。' },
        confidence: confidence('low', 'confirm_candidates', ['JP', 'CN', 'US']),
        confidenceShortcut: 'candidates',
        confidenceCandidates: [
          { target: japan, score: 0.7 },
          { target: { level: 'country', id: 'CN', label: '中国', country_iso: 'CN' }, score: 0.69 },
          { target: { level: 'country', id: 'US', label: '美国', country_iso: 'US' }, score: 0.5 },
        ],
        locationConfirmationReady: true,
      },
    )

    expect(html).toContain('候选仅供参考')
    expect(html).toContain('仅在能够确认时选择')
    expect(html).toContain('线索不足时的候选，请仅在能够确认时选择。')
    expect(html.match(/guided-confidence-card/g)).toHaveLength(3)
    expect(html.match(/class="mini-btn confirm"/g)).toHaveLength(3)
    expect(html).toContain('这些只是线索不足时的可选候选；仅在能够确认时选择，未确认前不会推进。')
  })

  it('does not inherit a country confidence shortcut below country stage', async () => {
    const html = await render(
      dialogue({
        stage: 'city',
        confirmed: { country: japan, admin1: null, city: null },
      }),
      [],
      {
        action: { kind: 'result', message: '不应继承的国家结果。' },
        confidence: confidence('high', 'direct_top1', ['JP', 'CN', 'US']),
        confidenceShortcut: 'high',
        confidenceCandidates: [{ target: japan, score: 0.9 }],
        locationConfirmationReady: true,
      },
    )

    expect(html).not.toContain('推荐国家 / 地区')
    expect(html).not.toContain('不应继承的国家结果')
    expect(html).not.toContain('guided-confidence-card')
  })
})

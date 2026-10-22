import { createSSRApp, type Component } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import ChatLog from '@/components/ChatLog.vue'
import ClueList from '@/components/ClueList.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import ImageViewer from '@/components/ImageViewer.vue'
import type { Clue } from '@/types'
import { clueIdentity } from '@/utils/clues'

function render(component: Component, props: Record<string, unknown>): Promise<string> {
  return renderToString(createSSRApp(component, props))
}

const signage: Clue[] = Array.from({ length: 6 }, (_, index) => ({
  dimension: 'signage',
  value: `出口 ${index}`,
  confidence: 0.7,
  evidence: `OCR 文本 ${index}`,
  source: 'ocr',
}))

const clueFixture: Clue[] = [
  { dimension: 'language', value: 'ja', confidence: 0.9, source: 'user', evidence: '用户确认' },
  ...signage,
  { dimension: 'mystery_dimension', value: 'mystery-value', confidence: 0.4 },
]

describe('ClueList rendering (SSR)', () => {
  it('renders localized groups, summary counts and capped OCR lines with expand', async () => {
    const html = await render(ClueList, { clues: clueFixture })
    expect(html).toContain('文字语言')
    expect(html).toContain('招牌文字')
    expect(html).toContain('mystery_dimension')
    expect(html).toMatch(/<strong[^>]*>3<\/strong> 个维度/)
    expect(html).toMatch(/<strong[^>]*>8<\/strong> 条线索/)
    expect(html).toContain('展开其余 2 条')
    expect(html).toContain('你补充的')
    expect(html).toContain('图片文字')
    expect(html).toContain('识别置信度 70%')
    expect(html).toContain('线索置信度 90%')
    expect(html).toContain('不是地点概率')
  })

  it('marks reference-only clues and catalog-validated clues distinctly', async () => {
    const html = await render(ClueList, {
      clues: [
        {
          dimension: 'road',
          value: 'street',
          confidence: 0.5,
          source: 'semantic',
          extra: { display_only: true, affects_ranking: false },
        },
        {
          dimension: 'place_name',
          value: 'JP',
          confidence: 0.8,
          source: 'semantic',
          extra: { validated: true },
        },
      ],
    })
    expect(html).toContain('仅供参考')
    expect(html).toContain('目录已核对')
  })

  it('shows the empty hint when no clues arrive', async () => {
    const html = await render(ClueList, { clues: [] })
    expect(html).toContain('上传照片后会自动识别线索。')
    expect(html).not.toContain('展开其余')
  })

  it('auto-expands and marks a newly appended tail clue in an existing 5+ group', async () => {
    const clues: Clue[] = Array.from({ length: 5 }, (_, index) => ({
      dimension: 'signage',
      value: `原始观察 ${index + 1}`,
      confidence: 0.9 - index * 0.1,
      evidence: `原始证据 ${index + 1}`,
      source: 'ocr',
    }))
    const appended: Clue = {
      dimension: 'signage',
      value: '尾部新增观察',
      confidence: 0.1,
      evidence: '对话追加的新证据',
      source: 'user',
    }
    const html = await render(ClueList, {
      clues: [...clues, appended],
      recentClueKeys: [clueIdentity(appended.dimension, appended.value)],
    })

    expect(html).toContain('尾部新增观察')
    expect(html).toContain('对话追加的新证据')
    expect(html).toContain('新线索')
    expect(html).toMatch(/class="[^"]*is-new[^"]*"/)
    expect(html).not.toContain('展开其余')
  })

  it('keeps short evidence visible and puts long evidence behind details', async () => {
    const html = await render(ClueList, {
      clues: [
        {
          dimension: 'signage',
          value: '短证据线索',
          confidence: 0.8,
          source: 'ocr',
          evidence: '这是一条简短证据',
        },
        {
          dimension: 'signage',
          value: '长证据线索',
          confidence: 0.7,
          source: 'ocr',
          evidence: '这是一条需要折叠展示的长证据。'.repeat(10),
        },
      ],
    })

    expect(html).toMatch(/class="ev"[^>]*>这是一条简短证据/)
    expect(html).toContain('class="clue-evidence-details"')
    expect(html).toMatch(/<summary[^>]*>查看依据<\/summary>/)
  })

  it('offers source-region jumps only for valid backend boxes', async () => {
    const html = await render(ClueList, {
      clues: [
        {
          dimension: 'signage',
          value: '精确框线索',
          confidence: 0.8,
          source: 'ocr',
          extra: { bbox_norm: { x: 0.1, y: 0.2, w: 0.3, h: 0.15 } },
        },
        {
          dimension: 'road',
          value: '区域框线索',
          confidence: 0.6,
          source: 'semantic',
          extra: { region_bbox_norm: { x: 0.05, y: 0.1, w: 0.5, h: 0.4 } },
        },
        {
          dimension: 'vehicle',
          value: '无框线索',
          confidence: 0.5,
          source: 'model',
          extra: { bbox_norm: { x: 0.8, y: 0.8, w: 0.3, h: 0.3 } },
        },
      ],
    })

    expect(html.match(/class="locate-clue"/g)).toHaveLength(2)
    expect(html).toContain('在照片中查看')
  })
})

describe('ImageViewer evidence focus (SSR)', () => {
  const focusedClue: Clue = {
    dimension: 'signage',
    value: '出口',
    confidence: 0.82,
    source: 'ocr',
    extra: { bbox_norm: { x: 0.2, y: 0.25, w: 0.25, h: 0.12 }, region_id: 'ra_test' },
  }

  it('renders a visibly labelled source box for a focused clue without creating a selection', async () => {
    const html = await render(ImageViewer, {
      imageUrl: '/media/photo.jpg',
      analyzing: false,
      status: '',
      statusKind: 'idle',
      focusedClue,
    })

    expect(html).toContain('class="focused-clue-box"')
    expect(html).toContain('图片文字 · 来源区域')
    expect(html).toContain('线索：出口（图片文字）')
    expect(html).toContain('还没选择区域')
  })

  it('does not render source geometry when the focused clue has no real box', async () => {
    const html = await render(ImageViewer, {
      imageUrl: '/media/photo.jpg',
      analyzing: false,
      status: '',
      statusKind: 'idle',
      focusedClue: { ...focusedClue, extra: { region_id: 'ra_without_geometry' } },
    })

    expect(html).not.toContain('class="focused-clue-box"')
    expect(html).not.toContain('来源区域')
  })
})

describe('FeedbackBar rendering (SSR)', () => {
  it('renders structured answer chips with localized dimension names and labels', async () => {
    const html = await render(FeedbackBar, {
      action: {
        kind: 'ask',
        message: '请问该地点的气候更符合哪一种特征？',
        payload: {
          dimensions: ['climate'],
          options: [
            { dimension: 'climate', value: 'arid', label: '干旱气候' },
            { dimension: 'language', value: 'ja' },
          ],
        },
      },
      busy: false,
      disabled: false,
    })
    expect(html).toContain('气候特征')
    expect(html).toContain('文字语言')
    expect(html).toContain('干旱气候')
    expect(html).toContain('日语')
    expect(html).toContain('option-chip')
    expect(html).toContain('<textarea')
    expect(html).toContain('补充线索（气候特征）')
  })

  it('keeps confirm/reject affordances for confirm actions', async () => {
    const html = await render(FeedbackBar, {
      action: { kind: 'confirm', payload: { iso: 'JP', name_zh: '日本' } },
      busy: false,
    })
    expect(html).toContain('确认这个国家')
    expect(html).toContain('不是这里')
    expect(html).not.toContain('option-chip')
  })

  it('renders confidence tier and direct-result state', async () => {
    const confidence = {
      tier: 'high',
      action: 'direct_top1',
      model_only_margin: 0.03,
      fused_margin: 0.02,
      effective_margin: 0.02,
      prior_top_iso: 'JP',
      top1_iso: 'JP',
      top3: [],
      leader_changed: false,
      reason: 'stable',
      policy_version: 'test',
    }
    const html = await render(FeedbackBar, {
      action: { kind: 'result', payload: { iso: 'JP', confidence } },
      busy: false,
    })
    expect(html).toContain('已找到最可能的国家，请确认。')
    expect(html).toContain('已给出最可能的国家，仍需你确认')
  })
})

describe('ChatLog rendering (SSR)', () => {
  it('renders system and AI messages with role classes', async () => {
    const html = await render(ChatLog, {
      messages: [
        { id: 1, role: 'system', content: '会话已恢复' },
        { id: 2, role: 'ai', content: '请补充可辨识线索' },
      ],
    })
    expect(html).toContain('class="system msg"')
    expect(html).toContain('会话已恢复')
    expect(html).toContain('请补充可辨识线索')
  })
})

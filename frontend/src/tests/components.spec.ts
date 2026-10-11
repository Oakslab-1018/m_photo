import { createSSRApp, type Component } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import ChatLog from '@/components/ChatLog.vue'
import ClueList from '@/components/ClueList.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import type { Clue } from '@/types'

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
    expect(html).toMatch(/<strong>3<\/strong> 个维度/)
    expect(html).toMatch(/<strong>8<\/strong> 条线索/)
    expect(html).toContain('展开其余 2 条')
    expect(html).toContain('用户补充')
    expect(html).toContain('OCR 识别')
    expect(html).toContain('置信度 70%')
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
    expect(html).toContain('也可在对话中手动补充')
    expect(html).not.toContain('展开其余')
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
    expect(html).toContain('我来补充（气候特征）')
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
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

    expect(html).toContain('已排除证据')
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
})
describe('guided audit text', () => {
  it('does not expose backend stage identifiers or rewrite location labels', () => {
    expect(safeGuidedReceipt('已回退到admin1阶段。')).toContain('省份 / 地区')
    expect(safeGuidedText('已确认城市：Mexico City。')).toBe('已确认城市：Mexico City。')
    expect(safeGuidedText('已回退到city阶段。')).toContain('城市')
  })
})


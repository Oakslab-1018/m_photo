import { nextTick, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import {
  useGuidedComposer,
  type GuidedFreeTextSubmission,
} from '@/composables/useGuidedComposer'

function setup(initialSuggestion: string | null = null) {
  const suggested = ref<string | null>(initialSuggestion)
  const submission = ref<GuidedFreeTextSubmission | null>(null)
  const busy = ref(false)
  const send = vi.fn<(text: string) => void>()
  const composer = useGuidedComposer({
    suggestedText: () => suggested.value,
    submission: () => submission.value,
    isBusy: () => busy.value,
    onSend: send,
  })
  return { suggested, submission, busy, send, composer }
}

describe('guided conversation composer', () => {
  it('sends one trimmed conversation turn with the supported example text', () => {
    const { send, composer } = setup()
    composer.text.value = '  路牌上是日文，车辆靠左行驶  '

    expect(composer.canSend.value).toBe(true)
    expect(composer.submit()).toBe(true)
    expect(send).toHaveBeenCalledTimes(1)
    expect(send).toHaveBeenCalledWith('路牌上是日文，车辆靠左行驶')
  })

  it('retains the draft through pending, stale and failed requests and clears only on matching success', async () => {
    const { submission, composer } = setup()
    const text = '路牌上是日文，车辆靠左行驶'
    composer.text.value = text
    composer.submit()

    submission.value = { text, status: 'pending' }
    await nextTick()
    expect(composer.text.value).toBe(text)

    submission.value = {
      text,
      status: 'failed',
      error: '服务端状态已更新，原操作未应用。',
    }
    await nextTick()
    expect(composer.text.value).toBe(text)

    submission.value = {
      text,
      status: 'unconfirmed',
      error: '服务器未返回可识别的回执。',
    }
    await nextTick()
    expect(composer.text.value).toBe(text)

    submission.value = { text, status: 'applied' }
    await nextTick()
    expect(composer.text.value).toBe('')
  })

  it('clears an already-saved clue without reporting a failed retry', async () => {
    const { submission, composer } = setup()
    const text = '玉林路'
    composer.text.value = text
    composer.submit()

    submission.value = { text, status: 'unchanged' }
    await nextTick()
    expect(composer.text.value).toBe('')

    composer.text.value = '另一条线索'
    submission.value = { text, status: 'failed', error: '没有识别到新的线索' }
    await nextTick()
    expect(composer.text.value).toBe('另一条线索')
  })

  it('never overwrites or clears a newer draft with an old submission acknowledgement', async () => {
    const { submission, composer } = setup()
    composer.text.value = '旧线索'
    composer.submit()

    composer.text.value = '新线索仍在编辑'
    submission.value = { text: '旧线索', status: 'failed', error: '失败' }
    await nextTick()
    expect(composer.text.value).toBe('新线索仍在编辑')

    submission.value = { text: '旧线索', status: 'applied' }
    await nextTick()
    expect(composer.text.value).toBe('新线索仍在编辑')
  })

  it('honors IME, busy guards and Enter submission semantics', () => {
    const { busy, send, composer } = setup()
    const preventWhileComposing = vi.fn()
    composer.text.value = '日文路牌'

    composer.composing.value = true
    expect(composer.submit()).toBe(false)
    composer.onEnter({
      isComposing: true,
      preventDefault: preventWhileComposing,
    } as unknown as KeyboardEvent)
    expect(preventWhileComposing).not.toHaveBeenCalled()
    expect(send).not.toHaveBeenCalled()

    composer.composing.value = false
    busy.value = true
    expect(composer.canSend.value).toBe(false)
    expect(composer.submit()).toBe(false)

    busy.value = false
    const prevent = vi.fn()
    composer.onEnter({ isComposing: false, preventDefault: prevent } as unknown as KeyboardEvent)
    expect(prevent).toHaveBeenCalledTimes(1)
    expect(send).toHaveBeenCalledWith('日文路牌')
  })

  it('accepts a one-time suggestion without treating its removal as a draft reset', async () => {
    const { suggested, composer } = setup('这个区域里的招牌文字不够清晰吗？')
    expect(composer.text.value).toBe('这个区域里的招牌文字不够清晰吗？')

    suggested.value = null
    await nextTick()
    expect(composer.text.value).toBe('这个区域里的招牌文字不够清晰吗？')
  })
})

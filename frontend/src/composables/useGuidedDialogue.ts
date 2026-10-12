import { computed, ref, watch, type Ref } from 'vue'
import { api } from '@/api'
import { ApiError, normalizeApiError } from '@/api/errors'
import type {
  GuidedDialogue,
  GuidedDialogueResponse,
  GuidedEventPayload,
  GuidedEventReceipt,
  GuidedLocationLevel,
  GuidedQuestion,
  GuidedTargetRef,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import { normalizeIsoCode } from '@/utils/geo'

export interface GuidedDialogueApi {
  dialogueStart(sessionId: number): Promise<GuidedDialogueResponse>
  dialogue(sessionId: number): Promise<GuidedDialogueResponse>
  dialogueEvent(sessionId: number, event: GuidedEventPayload): Promise<GuidedDialogueResponse>
}

export interface UseGuidedDialogueOptions {
  api?: GuidedDialogueApi
  onSnapshot?: (response: GuidedDialogueResponse) => void
  eventIdFactory?: () => string
}

/** Event body before the composable attaches idempotency and revision fields. */
export type GuidedEventInput = Omit<GuidedEventPayload, 'event_id' | 'expected_revision'>

let fallbackEventCounter = 0

function defaultEventId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID()
  fallbackEventCounter += 1
  return `guided-${Date.now().toString(36)}-${fallbackEventCounter}`
}

function errorStatus(error: unknown): number | undefined {
  if (error instanceof ApiError) return error.status
  return normalizeApiError(error).status
}

function isRetryableStatus(status: number | undefined): boolean {
  return status === undefined || status === 408 || status === 425 || status === 429 || status >= 500
}

function buildEvent(
  input: GuidedEventInput,
  revision: number,
  eventId: string,
): GuidedEventPayload {
  const event: GuidedEventPayload = {
    type: input.type,
    event_id: eventId,
    expected_revision: revision,
  }
  if (input.question_id !== undefined) event.question_id = input.question_id
  if (input.target) event.target = { level: input.target.level, id: input.target.id }
  if (input.clue_id !== undefined) event.clue_id = input.clue_id
  if (input.value !== undefined) event.value = input.value
  if (input.text !== undefined) event.text = input.text
  if (input.level !== undefined) event.level = input.level
  return event
}

function validateEvent(input: GuidedEventInput): string | null {
  switch (input.type) {
    case 'confirm_location':
    case 'reject_location':
      if (!input.target) return '请选择一个后端返回的地点候选。'
      return null
    case 'confirm_clue':
    case 'reject_clue':
      if (!input.clue_id?.trim()) return '缺少要核对的线索。'
      return null
    case 'answer':
      if (!input.question_id?.trim()) return '当前没有可回答的问题，请先重新加载。'
      if (!input.value?.trim()) return '请选择一个答案。'
      return null
    case 'unknown':
      if (!input.question_id?.trim()) return '当前没有可标记为不确定的问题。'
      return null
    case 'retract':
      if (!input.target && !input.clue_id?.trim()) return '缺少要撤回的排除项。'
      return null
    case 'backtrack':
      if (!input.level) return '缺少要返回的层级。'
      return null
    case 'free_text':
      if (!input.text?.trim()) return '请输入文字线索或目录名称。'
      return null
  }
}

/**
 * Authoritative guided-dialogue state and event transport.
 *
 * The composable never synthesizes a target/score/status. It only forwards
 * explicit UI events to the backend, applies returned snapshots, and keeps one
 * stable event_id for manual retries.
 */
export function useGuidedDialogue(
  sessionId: Ref<number | null>,
  options: UseGuidedDialogueOptions = {},
) {
  const client = options.api ?? {
    dialogueStart: (id: number) => api.dialogueStart(id),
    dialogue: (id: number) => api.dialogue(id),
    dialogueEvent: (id: number, event: GuidedEventPayload) => api.dialogueEvent(id, event),
  }
  const eventIdFactory = options.eventIdFactory ?? defaultEventId

  const active = ref(false)
  const loading = ref(false)
  const applying = ref(false)
  const error = ref<string | null>(null)
  const conflict = ref<string | null>(null)
  const dialogue = ref<GuidedDialogue | null>(null)
  const candidates = ref<LocationCandidate[]>([])
  const receipt = ref<GuidedEventReceipt | null>(null)
  const failedEvent = ref<GuidedEventPayload | null>(null)
  const failedQuestion = ref<GuidedQuestion | null>(null)
  const retryable = ref(false)
  const activeSessionId = ref<number | null>(null)

  const pendingQuestion = computed<GuidedQuestion | null>(
    () => dialogue.value?.pending_question ?? null,
  )
  const canRetry = computed(() => retryable.value && failedEvent.value !== null)

  let generation = 0

  function isCurrent(token: number, id: number): boolean {
    return token === generation && activeSessionId.value === id
  }

  function resetState(): void {
    active.value = false
    loading.value = false
    applying.value = false
    error.value = null
    conflict.value = null
    dialogue.value = null
    candidates.value = []
    receipt.value = null
    failedEvent.value = null
    failedQuestion.value = null
    retryable.value = false
    activeSessionId.value = null
  }

  function clear(): void {
    generation += 1
    resetState()
  }

  watch(
    sessionId,
    (id) => {
      if (id !== activeSessionId.value) clear()
    },
    { flush: 'sync' },
  )

  function applyResponse(response: GuidedDialogueResponse): void {
    if (!response?.dialogue || typeof response.dialogue.revision !== 'number') {
      throw new ApiError({ message: '后端返回的引导状态不完整，请重试' })
    }
    dialogue.value = response.dialogue
    candidates.value = [...(response.candidates ?? [])]
    receipt.value = response.receipt ?? null
    options.onSnapshot?.(response)
  }

  async function start(id?: number | null): Promise<boolean> {
    const target = id ?? activeSessionId.value ?? sessionId.value
    if (target === null || target === undefined) {
      error.value = '当前没有会话，请先上传图片。'
      return false
    }

    const token = ++generation
    activeSessionId.value = target
    active.value = true
    loading.value = true
    applying.value = false
    error.value = null
    conflict.value = null
    dialogue.value = null
    candidates.value = []
    receipt.value = null
    failedEvent.value = null
    failedQuestion.value = null
    retryable.value = false

    try {
      const response = await client.dialogueStart(target)
      if (!isCurrent(token, target)) return false
      applyResponse(response)
      return true
    } catch (caught) {
      if (!isCurrent(token, target)) return false
      error.value = normalizeApiError(caught, undefined, '引导定位启动失败，请重试').message
      return false
    } finally {
      if (isCurrent(token, target)) loading.value = false
    }
  }

  async function reload(keepFailure = false): Promise<boolean> {
    const target = activeSessionId.value ?? sessionId.value
    if (target === null || target === undefined) return false

    const token = ++generation
    loading.value = true
    applying.value = false
    if (!keepFailure) {
      conflict.value = null
      error.value = null
      failedEvent.value = null
      failedQuestion.value = null
      retryable.value = false
    }

    try {
      const response = await client.dialogue(target)
      if (!isCurrent(token, target)) return false
      applyResponse(response)
      if (!keepFailure) {
        error.value = null
        conflict.value = null
      }
      return true
    } catch (caught) {
      if (!isCurrent(token, target)) return false
      error.value = normalizeApiError(caught, undefined, '无法读取引导状态，请重试').message
      return false
    } finally {
      if (isCurrent(token, target)) loading.value = false
    }
  }

  function sameTarget(left: GuidedTargetRef, right: GuidedTargetRef): boolean {
    return left.level === right.level && left.id === right.id
  }

  function hasCandidate(target: GuidedTargetRef): boolean {
    return candidates.value.some((candidate) => sameTarget(candidate.target, target))
  }

  function hasExclusion(target: GuidedTargetRef): boolean {
    return (dialogue.value?.exclusions ?? []).some((item) => sameTarget(item, target))
  }

  /**
   * A 409 means no mutation happened, but the refreshed snapshot may no longer
   * contain the question/target/clue the failed event referred to. Only a
   * matching state may reuse the retained event_id for a deliberate retry.
   */
  function canRetryFailedEvent(
    event: GuidedEventPayload,
    questionContext: GuidedQuestion | null,
  ): boolean {
    const current = dialogue.value
    if (!current) return false

    if (event.question_id) {
      const pending = current.pending_question
      if (!pending || pending.id !== event.question_id) return false
      if (questionContext && pending.kind !== questionContext.kind) return false
      if (questionContext?.clue_id && pending.clue_id !== questionContext.clue_id) return false
      if (event.clue_id && pending.clue_id !== event.clue_id) return false

      if (event.type === 'confirm_location' || event.type === 'reject_location') {
        if (pending.kind !== 'confirm_location') return false
      }
      if (event.type === 'confirm_clue' || event.type === 'reject_clue') {
        if (pending.kind !== 'review_clue' || pending.clue_id !== event.clue_id) return false
      }
      if (event.type === 'answer') {
        if (!event.value?.trim()) return false
        const options = pending.options ?? []
        if (options.length && !options.some((option) => option.value === event.value)) return false
      }
      if (event.target && !hasCandidate(event.target)) return false
      return true
    }

    if (event.target) {
      return event.type === 'retract' ? hasExclusion(event.target) : hasCandidate(event.target)
    }
    if (event.clue_id) {
      const clue = current.clues.find((item) => item.id === event.clue_id)
      if (!clue) return false
      if (event.type === 'confirm_clue' || event.type === 'reject_clue') return clue.status === 'pending'
      if (event.type === 'retract') return clue.status !== 'pending'
    }
    return true
  }

  function discardFailedEvent(message: string): void {
    failedEvent.value = null
    failedQuestion.value = null
    retryable.value = false
    conflict.value = null
    error.value = message
  }
  async function performEvent(event: GuidedEventPayload, questionContext: GuidedQuestion | null = null): Promise<boolean> {
    const target = activeSessionId.value ?? sessionId.value
    if (target === null || target === undefined) return false
    if (applying.value || loading.value) return false

    const token = ++generation
    applying.value = true
    error.value = null
    conflict.value = null

    try {
      const response = await client.dialogueEvent(target, event)
      if (!isCurrent(token, target)) return false
      applyResponse(response)
      failedEvent.value = null
      failedQuestion.value = null
      retryable.value = false
      return true
    } catch (caught) {
      if (!isCurrent(token, target)) return false
      applying.value = false
      const status = errorStatus(caught)

      if (status === 409) {
        conflict.value = '服务端状态已更新，已重新加载；请核对最新引导后手动重试。'
        failedEvent.value = event
        failedQuestion.value = questionContext
        retryable.value = true
        // Never replay automatically: reload and require a matching fresh state.
        const reloaded = await reload(true)
        if (!reloaded || !canRetryFailedEvent(event, questionContext)) {
          discardFailedEvent(
            reloaded
              ? '引导问题已更新，原操作已丢弃；请按最新问题重新操作。'
              : '无法确认最新引导状态，原操作已丢弃；请重新加载后再试。',
          )
        }
        return false
      }

      if (status === 422) {
        error.value = '当前操作与最新引导状态不一致，请按最新选项重新操作。'
        failedEvent.value = null
        failedQuestion.value = null
        retryable.value = false
        return false
      }

      error.value = normalizeApiError(caught, undefined, '引导操作失败，请稍后重试').message
      failedEvent.value = event
      failedQuestion.value = questionContext
      retryable.value = isRetryableStatus(status)
      return false
    } finally {
      if (isCurrent(token, target)) applying.value = false
    }
  }

  function send(input: GuidedEventInput): Promise<boolean> {
    const current = dialogue.value
    if (!active.value || !current) {
      error.value = '引导状态尚未就绪，请先重新加载。'
      return Promise.resolve(false)
    }

    const invalid = validateEvent(input)
    if (invalid) {
      error.value = invalid
      return Promise.resolve(false)
    }

    const event = buildEvent(input, current.revision, eventIdFactory())
    const questionContext = event.question_id ? current.pending_question : null
    return performEvent(event, questionContext)
  }

  function questionIdFor(clueId?: string): string | undefined {
    const question = dialogue.value?.pending_question
    if (!question) return undefined
    if (clueId && question.clue_id !== clueId) return undefined
    return question.id
  }

  function locationQuestionId(): string | undefined {
    const question = dialogue.value?.pending_question
    return question?.kind === 'confirm_location' ? question.id : undefined
  }

  function targetRef(target: LocationTarget): GuidedTargetRef {
    return { level: target.level, id: target.id }
  }

  function confirmLocation(target: LocationTarget): Promise<boolean> {
    return send({
      type: 'confirm_location',
      target: targetRef(target),
      question_id: locationQuestionId(),
    })
  }

  function rejectLocation(target: LocationTarget): Promise<boolean> {
    return send({
      type: 'reject_location',
      target: targetRef(target),
      question_id: locationQuestionId(),
    })
  }

  function targetForCountry(iso: string): LocationTarget | null {
    const code = normalizeIsoCode(iso)
    if (!code) return null
    const candidate = candidates.value.find((item) => {
      if (item.target.level !== 'country') return false
      return [item.target.id, item.target.country_iso, item.target.source_iso]
        .some((value) => normalizeIsoCode(value ?? '') === code)
    })
    return candidate?.target ?? null
  }

  function confirmCountry(iso: string): Promise<boolean> {
    const target = targetForCountry(iso)
    if (!target) {
      error.value = '该国家 / 地区不在当前引导候选中，请使用主引导面板的服务器候选。'
      return Promise.resolve(false)
    }
    return confirmLocation(target)
  }

  function rejectCountry(iso: string): Promise<boolean> {
    const target = targetForCountry(iso)
    if (!target) {
      error.value = '该国家 / 地区不在当前引导候选中，请使用主引导面板的服务器候选。'
      return Promise.resolve(false)
    }
    return rejectLocation(target)
  }

  function confirmClue(clueId: string): Promise<boolean> {
    return send({
      type: 'confirm_clue',
      clue_id: clueId,
      question_id: questionIdFor(clueId),
    })
  }

  function rejectClue(clueId: string): Promise<boolean> {
    return send({
      type: 'reject_clue',
      clue_id: clueId,
      question_id: questionIdFor(clueId),
    })
  }

  function answer(value: string, questionId?: string): Promise<boolean> {
    const id = questionId ?? dialogue.value?.pending_question?.id
    if (!id) {
      error.value = '当前没有可回答的问题。'
      return Promise.resolve(false)
    }
    return send({ type: 'answer', value, question_id: id })
  }

  function unknown(questionId?: string): Promise<boolean> {
    const id = questionId ?? dialogue.value?.pending_question?.id
    if (!id) {
      error.value = '当前没有可标记为不确定的问题。'
      return Promise.resolve(false)
    }
    return send({ type: 'unknown', question_id: id })
  }

  function retractTarget(target: LocationTarget): Promise<boolean> {
    return send({ type: 'retract', target: targetRef(target) })
  }

  function retractClue(clueId: string): Promise<boolean> {
    return send({ type: 'retract', clue_id: clueId })
  }

  function backtrack(level: GuidedLocationLevel): Promise<boolean> {
    return send({ type: 'backtrack', level })
  }

  function freeText(text: string): Promise<boolean> {
    return send({ type: 'free_text', text: text.trim() })
  }

  /**
   * Manual retry preserves the original event_id and refreshes expected_revision.
   *
   * A 409 is defined as no mutation, so this is a deliberate retry against the
   * reloaded snapshot. If the backend did persist the original event but the
   * response was lost, it must return the stored receipt for the repeated
   * event_id before applying the revision precondition.
   */
  async function retry(): Promise<boolean> {
    if (applying.value) return false
    const event = failedEvent.value
    if (!event || !retryable.value) {
      if (dialogue.value) return reload()
      return start()
    }

    let revision = dialogue.value?.revision
    if (revision === undefined) {
      const reloaded = await reload(true)
      if (!reloaded) {
        discardFailedEvent('无法确认最新引导状态，原操作已丢弃；请重新加载后再试。')
        return false
      }
      revision = dialogue.value?.revision
    }
    if (revision === undefined) return false

    if (!canRetryFailedEvent(event, failedQuestion.value)) {
      discardFailedEvent('引导问题已更新，原操作已丢弃；请按最新问题重新操作。')
      return false
    }
    return performEvent({ ...event, expected_revision: revision }, failedQuestion.value)
  }

  return {
    active,
    loading,
    applying,
    error,
    conflict,
    dialogue,
    candidates,
    receipt,
    pendingQuestion,
    canRetry,
    start,
    reload,
    retry,
    clear,
    send,
    confirmLocation,
    rejectLocation,
    confirmCountry,
    rejectCountry,
    confirmClue,
    rejectClue,
    answer,
    unknown,
    retractTarget,
    retractClue,
    backtrack,
    freeText,
  }
}





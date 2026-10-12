import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { ApiError } from '@/api/errors'
import { useGeoSession } from '@/composables/useGeoSession'
import { useGuidedDialogue, type GuidedDialogueApi } from '@/composables/useGuidedDialogue'
import type {
  GuidedDialogue,
  GuidedDialogueResponse,
  LocationCandidate,
  LocationTarget,
  ReviewedClue,
} from '@/types'

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
    revision: 1,
    stage: 'country',
    confirmed: { country: null, admin1: null, city: null },
    exclusions: [],
    clues: [],
    pending_question: null,
    ...overrides,
  }
}

function snapshot(
  value: GuidedDialogue = dialogue(),
  candidates: LocationCandidate[] = [],
): GuidedDialogueResponse {
  return {
    dialogue: value,
    candidates,
    action: null,
    messages: [],
    scores: [],
    clues: [],
    exclusions: [],
  }
}

function apiMock(): GuidedDialogueApi & {
  dialogueStart: ReturnType<typeof vi.fn>
  dialogue: ReturnType<typeof vi.fn>
  dialogueEvent: ReturnType<typeof vi.fn>
} {
  return {
    dialogueStart: vi.fn(async () => snapshot()),
    dialogue: vi.fn(async () => snapshot()),
    dialogueEvent: vi.fn(async () => snapshot()),
  }
}

describe('useGuidedDialogue', () => {
  it('starts from the authoritative snapshot and applies legacy evidence only', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const onSnapshot = vi.fn()
    const initial = dialogue({ revision: 4 })
    api.dialogueStart.mockResolvedValue(snapshot(initial, [{ target: japan }]))
    const guided = useGuidedDialogue(sessionId, {
      api,
      onSnapshot,
      eventIdFactory: () => 'event-fixed',
    })

    await guided.start()

    expect(api.dialogueStart).toHaveBeenCalledWith(7)
    expect(guided.active.value).toBe(true)
    expect(guided.loading.value).toBe(false)
    expect(guided.dialogue.value?.revision).toBe(4)
    expect(guided.candidates.value.map((item) => item.target.id)).toEqual(['JP'])
    expect(onSnapshot).toHaveBeenCalledTimes(1)
  })

  it('sends expected_revision, a stable event id and question_id for answers', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const initial = dialogue({
      revision: 8,
      pending_question: {
        id: 'question-1',
        kind: 'ask',
        message: '气候更接近哪一种？',
        options: [{ value: 'arid', label: '干旱气候' }],
      },
    })
    api.dialogueStart.mockResolvedValue(snapshot(initial))
    api.dialogueEvent.mockResolvedValue(snapshot(dialogue({ revision: 9 })))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'event-answer',
    })

    await guided.start()
    await guided.answer('arid')

    expect(api.dialogueEvent).toHaveBeenCalledWith(7, {
      type: 'answer',
      event_id: 'event-answer',
      expected_revision: 8,
      question_id: 'question-1',
      value: 'arid',
    })
  })

  it('references the confirm_location question only for the exact pending target', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const canada: LocationTarget = {
      level: 'country',
      id: 'CA',
      label: '加拿大',
      country_iso: 'CA',
    }
    const question = {
      id: 'confirm-country-jp',
      kind: 'confirm_location' as const,
      message: '是否确认日本？',
      target: japan,
    }
    const candidates = [
      { target: japan, score: 0.8 },
      { target: canada, score: 0.7 },
    ]
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({ revision: 4, pending_question: question }),
      candidates,
    ))
    api.dialogueEvent.mockResolvedValue(snapshot(
      dialogue({ revision: 4, pending_question: question }),
      candidates,
    ))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'location-reference',
    })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    await guided.confirmLocation(canada)
    await guided.rejectLocation(canada)
    await guided.confirmLocation(japan)

    const events = api.dialogueEvent.mock.calls.map((call) => call[1])
    expect(events[0]).not.toHaveProperty('question_id')
    expect(events[1]).not.toHaveProperty('question_id')
    expect(events[2]?.question_id).toBe('confirm-country-jp')
    expect(events[0]?.target).toEqual({ level: 'country', id: 'CA' })
    expect(events[2]?.target).toEqual({ level: 'country', id: 'JP' })
  })

  it('prevents duplicate submission while one event is applying', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 3 }), [{ target: japan }]))
    let resolveEvent!: (value: GuidedDialogueResponse) => void
    api.dialogueEvent.mockReturnValueOnce(
      new Promise<GuidedDialogueResponse>((resolve) => {
        resolveEvent = resolve
      }),
    )
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'event-once',
    })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    const first = guided.confirmLocation(japan)
    const second = guided.confirmLocation(japan)

    expect(await second).toBe(false)
    expect(api.dialogueEvent).toHaveBeenCalledTimes(1)
    resolveEvent(snapshot(dialogue({ revision: 4 })))
    await first
    expect(guided.applying.value).toBe(false)
  })

  it('invalidates an older in-flight event when point reconciliation reloads state', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 1 }), [{ target: japan }]))
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({ revision: 2 }), [{ target: japan }]))
    let resolveEvent!: (value: GuidedDialogueResponse) => void
    api.dialogueEvent.mockReturnValueOnce(
      new Promise<GuidedDialogueResponse>((resolve) => {
        resolveEvent = resolve
      }),
    )
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'event-old',
    })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    const olderEvent = guided.confirmLocation(japan)
    await guided.reload()
    resolveEvent(snapshot(dialogue({ revision: 3 }), [{ target: japan }]))
    await olderEvent

    expect(guided.dialogue.value?.revision).toBe(2)
  })
  it('reloads on 409 and reuses the event id only for a manual retry', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 10 }), [{ target: japan }]))
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({ revision: 11 }), [{ target: japan }]))
    api.dialogueEvent
      .mockRejectedValueOnce(new ApiError({ message: 'conflict', status: 409 }))
      .mockResolvedValueOnce(snapshot(dialogue({ revision: 12 }), [{ target: japan }]))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'event-conflict',
    })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    const applied = await guided.confirmLocation(japan)

    expect(applied).toBe(false)
    expect(api.dialogue).toHaveBeenCalledWith(7)
    expect(guided.dialogue.value?.revision).toBe(11)
    expect(guided.conflict.value).toContain('已重新加载')
    expect(guided.canRetry.value).toBe(true)

    await guided.retry()

    const calls = api.dialogueEvent.mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[0][1].event_id).toBe('event-conflict')
    expect(calls[1][1].event_id).toBe('event-conflict')
    expect(calls[0][1].expected_revision).toBe(10)
    expect(calls[1][1].expected_revision).toBe(11)
  })

  it('discards a 409 retry when the refreshed question no longer matches', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 4,
      pending_question: {
        id: 'question-old',
        kind: 'ask',
        message: '旧问题',
        options: [{ value: 'yes', label: '是' }],
      },
    })))
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({
      revision: 5,
      pending_question: {
        id: 'question-new',
        kind: 'ask',
        message: '新问题',
        options: [{ value: 'no', label: '不是' }],
      },
    })))
    api.dialogueEvent.mockRejectedValueOnce(new ApiError({ message: 'conflict', status: 409 }))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'answer-stale',
    })

    await guided.start()
    const applied = await guided.answer('yes')

    expect(applied).toBe(false)
    expect(api.dialogueEvent).toHaveBeenCalledTimes(1)
    expect(guided.canRetry.value).toBe(false)
    expect(guided.error.value).toContain('最新问题')

    await guided.retry()
    expect(api.dialogueEvent).toHaveBeenCalledTimes(1)
  })

  it('reuses the event id when the refreshed question still matches', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const question = {
      id: 'question-stable',
      kind: 'ask' as const,
      message: '气候更接近哪一种？',
      options: [{ value: 'arid', label: '干旱气候' }],
    }
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 4, pending_question: question })))
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({ revision: 5, pending_question: question })))
    api.dialogueEvent
      .mockRejectedValueOnce(new ApiError({ message: 'conflict', status: 409 }))
      .mockResolvedValueOnce(snapshot(dialogue({ revision: 6 })))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'answer-stable',
    })

    await guided.start()
    await guided.answer('arid')
    expect(guided.canRetry.value).toBe(true)

    await guided.retry()
    const calls = api.dialogueEvent.mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[0][1].event_id).toBe('answer-stable')
    expect(calls[1][1].event_id).toBe('answer-stable')
    expect(calls[0][1].expected_revision).toBe(4)
    expect(calls[1][1].expected_revision).toBe(5)
  })
  it('routes a legacy country code through the exact backend candidate target', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 2 }), [{ target: japan }]))
    api.dialogueEvent.mockResolvedValue(snapshot(dialogue({ revision: 3 })))
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => 'event-country',
    })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    await guided.confirmCountry('jp')

    expect(api.dialogueEvent).toHaveBeenCalledWith(7, {
      type: 'confirm_location',
      event_id: 'event-country',
      expected_revision: 2,
      target: { level: 'country', id: 'JP' },
    })
  })

  it('maps clue review, unknown, retract and backtrack to explicit events', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const initial = dialogue({
      revision: 5,
      clues: [
        {
          id: 'clue-1',
          dimension: 'language',
          value: 'ja',
          description: '日语招牌',
          source: 'model',
          confidence: 0.8,
          status: 'pending',
        },
      ],
      exclusions: [japan],
      pending_question: {
        id: 'q-clue',
        kind: 'review_clue',
        clue_id: 'clue-1',
        message: '这条观察可靠吗？',
      },
    })
    api.dialogueStart.mockResolvedValue(snapshot(initial))
    api.dialogueEvent.mockResolvedValue(snapshot(initial))
    let eventNumber = 0
    const guided = useGuidedDialogue(sessionId, {
      api,
      eventIdFactory: () => `event-${++eventNumber}`,
    })

    await guided.start()
    await guided.confirmClue('clue-1')
    await guided.unknown()
    await guided.retractTarget(japan)
    await guided.retractClue('clue-1')
    await guided.backtrack('admin1')
    await guided.freeText('  浙江  ')

    expect(api.dialogueEvent.mock.calls.map((call) => call[1])).toEqual([
      {
        type: 'confirm_clue',
        event_id: 'event-1',
        expected_revision: 5,
        clue_id: 'clue-1',
        question_id: 'q-clue',
      },
      {
        type: 'unknown',
        event_id: 'event-2',
        expected_revision: 5,
        question_id: 'q-clue',
      },
      {
        type: 'retract',
        event_id: 'event-3',
        expected_revision: 5,
        target: { level: 'country', id: 'JP' },
      },
      {
        type: 'retract',
        event_id: 'event-4',
        expected_revision: 5,
        clue_id: 'clue-1',
      },
      {
        type: 'backtrack',
        event_id: 'event-5',
        expected_revision: 5,
        level: 'admin1',
      },
      {
        type: 'free_text',
        event_id: 'event-6',
        expected_revision: 5,
        text: '浙江',
      },
    ])
  })

  it('clears all guided state when the session changes', async () => {
    const sessionId = ref<number | null>(1)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 5 }), [{ target: japan }]))
    const guided = useGuidedDialogue(sessionId, { api })

    await guided.start()
    sessionId.value = 2

    expect(guided.active.value).toBe(false)
    expect(guided.dialogue.value).toBeNull()
    expect(guided.candidates.value).toEqual([])
  })

  it('keeps rankings hidden and blocks location confirmation until all observations are handled', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 2,
      clues: [
        {
          id: 'pending-clue',
          dimension: 'building',
          value: 'generic',
          description: '建筑风格不明显',
          source: 'building',
          confidence: 0.6,
          status: 'pending',
        },
      ],
    }), [{ target: japan, score: 0.9 }]))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'blocked-location' })

    await guided.start()

    expect(guided.pendingClueCount.value).toBe(1)
    expect(guided.rankingsUnlocked.value).toBe(false)
    expect(guided.locationConfirmationReady.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
    expect(await guided.confirmLocation(japan)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()
    expect(guided.error.value).toContain('待核对观察')
  })

  it('treats confirm, reject and unknown statuses as handled and keeps unlock sticky for new observations', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const reviewed = [
      {
        id: 'confirmed-clue',
        dimension: 'language',
        value: 'ja',
        description: '日语文字',
        source: 'model',
        confidence: 0.8,
        status: 'confirmed' as const,
      },
      {
        id: 'rejected-clue',
        dimension: 'building',
        value: 'gothic',
        description: '并非哥特式',
        source: 'model',
        confidence: 0.6,
        status: 'rejected' as const,
      },
      {
        id: 'uncertain-clue',
        dimension: 'climate',
        value: 'temperate',
        description: '气候暂不确定',
        source: 'model',
        confidence: 0.5,
        status: 'uncertain' as const,
      },
    ]
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({ revision: 3, clues: reviewed }),
      [{ target: japan, score: 0.8 }],
    ))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'sticky-location' })

    await guided.start()
    // Rankings are unlocked server-side, but the explicit next step gates them.
    expect(guided.rankingsUnlocked.value).toBe(true)
    expect(guided.phase.value).toBe('review')
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.locationConfirmationReady.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
    expect(await guided.confirmLocation(japan)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()

    expect(guided.nextPhase()).toBe(true)
    expect(guided.candidatesVisible.value).toBe(true)
    expect(guided.locationConfirmationReady.value).toBe(true)
    expect(guided.topCandidates.value.map((item) => item.target.id)).toEqual(['JP'])

    const pendingNew = {
      id: 'new-pending-clue',
      dimension: 'signage',
      value: '出口',
      description: '新发现的观察',
      source: 'ocr',
      confidence: 0.7,
      status: 'pending' as const,
    }
    api.dialogue.mockResolvedValueOnce(snapshot(
      dialogue({ revision: 4, clues: [...reviewed, pendingNew] }),
      [{ target: japan, score: 0.8 }],
    ))
    await guided.reload()

    expect(guided.rankingsUnlocked.value).toBe(true)
    // A newly appended observation pulls the UI back into the single review box.
    expect(guided.phase.value).toBe('review')
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.locationConfirmationReady.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
    expect(guided.recentClueIds.value).toEqual(['new-pending-clue'])
    expect(await guided.confirmLocation(japan)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()

    api.dialogue.mockResolvedValueOnce(snapshot(
      dialogue({
        revision: 5,
        clues: [...reviewed, { ...pendingNew, status: 'confirmed' as const }],
      }),
      [{ target: japan, score: 0.8 }],
    ))
    await guided.reload()
    expect(guided.reviewReady.value).toBe(true)
    expect(guided.nextPhase()).toBe(true)
    expect(guided.locationConfirmationReady.value).toBe(true)
  })

  it('rebuilds sticky rankings only for handled machine evidence plus pending user rows', async () => {
    const userPendingApi = apiMock()
    userPendingApi.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 6,
      clues: [
        {
          id: 'handled-model',
          dimension: 'language',
          value: 'ja',
          description: '已确认日语文字',
          source: 'model',
          confidence: 0.8,
          status: 'confirmed',
        },
        {
          id: 'pending-user',
          dimension: 'place_name',
          value: '杭州',
          description: '用户新增地名观察',
          source: 'user',
          confidence: 0.9,
          status: 'pending',
        },
      ],
    }), [{ target: japan, score: 0.8 }]))
    const userPending = useGuidedDialogue(ref<number | null>(7), { api: userPendingApi })

    await userPending.start()
    expect(userPending.rankingsUnlocked.value).toBe(true)
    // A pending user observation keeps the review box in charge: no next step,
    // no candidates, no location confirmation.
    expect(userPending.phase.value).toBe('review')
    expect(userPending.topCandidates.value).toEqual([])
    expect(userPending.locationConfirmationReady.value).toBe(false)
    expect(userPending.nextPhase()).toBe(false)
    expect(userPending.error.value).toContain('待核对观察')
    expect(await userPending.confirmLocation(japan)).toBe(false)
    expect(userPendingApi.dialogueEvent).not.toHaveBeenCalled()

    const machinePendingApi = apiMock()
    machinePendingApi.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 7,
      clues: [
        {
          id: 'handled-model',
          dimension: 'language',
          value: 'ja',
          description: '已确认日语文字',
          source: 'model',
          confidence: 0.8,
          status: 'confirmed',
        },
        {
          id: 'pending-ocr',
          dimension: 'signage',
          value: '出口',
          description: '刷新后仍待核对的机器观察',
          source: 'ocr',
          confidence: 0.7,
          status: 'pending',
        },
      ],
    }), [{ target: japan, score: 0.8 }]))
    const machinePending = useGuidedDialogue(ref<number | null>(7), { api: machinePendingApi })

    await machinePending.start()
    expect(machinePending.rankingsUnlocked.value).toBe(false)
    expect(machinePending.topCandidates.value).toEqual([])
    expect(machinePending.nextPhase()).toBe(false)
  })

  it('recomputes the Top-5 on every snapshot and rejects a hidden sixth target', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const candidates: LocationCandidate[] = [
      { target: japan, score: 0.4 },
      { target: { level: 'country', id: 'US', label: '美国', country_iso: 'US' }, score: 0.8 },
      { target: { level: 'country', id: 'FR', label: '法国', country_iso: 'FR' }, score: 0.2 },
      { target: { level: 'country', id: 'DE', label: '德国', country_iso: 'DE' }, score: 0.7 },
      { target: { level: 'country', id: 'GB', label: '英国', country_iso: 'GB' }, score: 0.1 },
      { target: { level: 'country', id: 'CA', label: '加拿大', country_iso: 'CA' }, score: 0.05 },
    ]
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({ revision: 1 }), candidates))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'top-update' })

    await guided.start()
    expect(guided.topCandidates.value).toEqual([])
    expect(guided.nextPhase()).toBe(true)
    expect(guided.topCandidates.value.map((item) => item.target.id)).toEqual([
      'US',
      'DE',
      'JP',
      'FR',
      'GB',
    ])
    const hidden = candidates[5]?.target
    expect(hidden).toBeTruthy()
    expect(await guided.confirmLocation(hidden as LocationTarget)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()

    api.dialogueEvent.mockResolvedValueOnce(snapshot(
      dialogue({ revision: 2 }),
      [
        ...candidates,
        { target: { level: 'country', id: 'IT', label: '意大利', country_iso: 'IT' }, score: 0.95 },
      ],
    ))
    await guided.confirmLocation(japan)
    expect(guided.topCandidates.value.map((item) => item.target.id)).toEqual([
      'IT',
      'US',
      'DE',
      'JP',
      'FR',
    ])
  })

  it('marks a blocking ask as unknown with the authoritative question id', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 2,
      pending_question: {
        id: 'ask-authoritative',
        kind: 'ask',
        message: '这里的气候更接近哪一种？',
        options: [{ value: 'arid', label: '干旱气候' }],
      },
    })))
    api.dialogueEvent.mockResolvedValue(snapshot(dialogue({ revision: 3 })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'unknown-ask' })

    await guided.start()
    expect(guided.reviewReady.value).toBe(false)

    expect(await guided.unknown('stale-question-id')).toBe(true)
    expect(api.dialogueEvent).toHaveBeenCalledWith(7, {
      type: 'unknown',
      event_id: 'unknown-ask',
      expected_revision: 2,
      question_id: 'ask-authoritative',
    })
    expect(guided.reviewHistory.value).toEqual([])
    expect(guided.reviewReady.value).toBe(true)

    // A location confirmation can never be skipped with 不确定.
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({
      revision: 4,
      pending_question: { id: 'confirm-1', kind: 'confirm_location', message: '确认地点？' },
    })))
    await guided.reload()
    expect(await guided.unknown()).toBe(false)
    expect(api.dialogueEvent).toHaveBeenCalledTimes(1)
  })

  it('tolerates snapshots whose clue rows omit the source field', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 1,
      clues: [
        {
          id: 'clue-1',
          dimension: 'language',
          value: 'zh',
          description: '缺少来源字段',
          status: 'pending',
        } as unknown as ReviewedClue,
      ],
      pending_question: {
        id: 'review-1',
        kind: 'review_clue',
        clue_id: 'clue-1',
        message: '这条观察准确吗？',
      },
    })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'missing-source' })

    await guided.start()

    expect(guided.active.value).toBe(true)
    expect(guided.error.value).toBeNull()
    expect(guided.rankingsUnlocked.value).toBe(false)
    expect(guided.currentReviewClue.value?.id).toBe('clue-1')
  })

  it('keeps every candidate entry hidden until the explicit next phase', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({
        revision: 2,
        clues: [
          {
            id: 'done-clue',
            dimension: 'language',
            value: 'ja',
            description: '已核对的观察',
            source: 'model',
            confidence: 0.8,
            status: 'confirmed',
          },
        ],
      }),
      [{ target: japan, score: 0.9 }],
    ))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'gate' })

    await guided.start()

    expect(guided.phase.value).toBe('review')
    expect(guided.reviewReady.value).toBe(true)
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
    expect(guided.locationConfirmationReady.value).toBe(false)
    expect(await guided.confirmLocation(japan)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()

    expect(guided.nextPhase()).toBe(true)
    expect(guided.phase.value).toBe('candidates')
    expect(guided.candidatesVisible.value).toBe(true)
    expect(guided.topCandidates.value.map((item) => item.target.id)).toEqual(['JP'])
    expect(guided.locationConfirmationReady.value).toBe(true)
  })

  it('hides the next step while a blocking ask is unresolved and reopens it after the answer', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 3,
      pending_question: {
        id: 'ask-1',
        kind: 'ask',
        message: '这段文字是什么？',
        options: [{ value: 'zh', label: '中文' }],
      },
    })))
    api.dialogueEvent.mockResolvedValue(snapshot(dialogue({ revision: 4 })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'ask-block' })

    await guided.start()
    expect(guided.reviewReady.value).toBe(false)
    expect(guided.nextPhase()).toBe(false)
    expect(guided.phase.value).toBe('review')
    expect(guided.error.value).toContain('阻塞')

    await guided.answer('zh')
    expect(guided.reviewReady.value).toBe(true)
    expect(guided.nextPhase()).toBe(true)
    expect(guided.phase.value).toBe('candidates')
  })

  it('pushes review history only on success and 返回上一条观察 retracts the pinned clue', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const pinned = {
      id: 'clue-1',
      dimension: 'language',
      value: 'ja',
      description: '日语招牌',
      source: 'model',
      confidence: 0.8,
      status: 'pending' as const,
    }
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 2,
      clues: [pinned],
      pending_question: {
        id: 'review-1',
        kind: 'review_clue',
        clue_id: 'clue-1',
        message: '这条观察准确吗？',
      },
    })))
    api.dialogueEvent
      .mockResolvedValueOnce(snapshot(dialogue({
        revision: 3,
        clues: [{ ...pinned, status: 'confirmed' }],
        pending_question: null,
      })))
      .mockResolvedValueOnce(snapshot(dialogue({
        revision: 4,
        clues: [{ ...pinned, status: 'pending' }],
        pending_question: {
          id: 'review-1b',
          kind: 'review_clue',
          clue_id: 'clue-1',
          message: '这条观察准确吗？',
        },
      })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'history' })

    await guided.start()
    expect(guided.reviewHistory.value).toEqual([])
    expect(guided.canReviewBack.value).toBe(false)

    await guided.confirmClue('clue-1')
    const events = api.dialogueEvent.mock.calls.map((call) => call[1])
    expect(events[0]).toMatchObject({ type: 'confirm_clue', clue_id: 'clue-1' })
    expect(events[0]?.question_id).toBe('review-1')
    expect(guided.reviewHistory.value).toEqual(['clue-1'])
    expect(guided.canReviewBack.value).toBe(true)
    expect(guided.reviewReady.value).toBe(true)

    expect(await guided.reviewBack()).toBe(true)
    const afterBack = api.dialogueEvent.mock.calls.map((call) => call[1])
    expect(afterBack[1]).toMatchObject({ type: 'retract', clue_id: 'clue-1' })
    expect(afterBack[1]).not.toHaveProperty('question_id')
    expect(guided.reviewHistory.value).toEqual([])
    expect(guided.canReviewBack.value).toBe(false)
    expect(guided.phase.value).toBe('review')
    expect(guided.currentReviewClue.value?.id).toBe('clue-1')
    expect(guided.reviewProgress.value).toEqual({ current: 1, total: 1 })
  })

  it('keeps review history when 返回上一条观察 hits a 409', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const pinned = {
      id: 'clue-1',
      dimension: 'language',
      value: 'ja',
      description: '日语招牌',
      source: 'model',
      confidence: 0.8,
      status: 'pending' as const,
    }
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 2,
      clues: [pinned],
      pending_question: {
        id: 'review-1',
        kind: 'review_clue',
        clue_id: 'clue-1',
        message: '这条观察准确吗？',
      },
    })))
    api.dialogueEvent
      .mockResolvedValueOnce(snapshot(dialogue({
        revision: 3,
        clues: [{ ...pinned, status: 'confirmed' }],
        pending_question: null,
      })))
      .mockRejectedValueOnce(new ApiError({ message: 'conflict', status: 409 }))
    api.dialogue.mockResolvedValueOnce(snapshot(dialogue({
      revision: 4,
      clues: [{ ...pinned, status: 'confirmed' }],
      pending_question: null,
    })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'retract-conflict' })

    await guided.start()
    await guided.confirmClue('clue-1')
    expect(guided.nextPhase()).toBe(true)
    expect(guided.phase.value).toBe('candidates')

    const applied = await guided.reviewBack()

    expect(applied).toBe(false)
    expect(api.dialogueEvent).toHaveBeenCalledTimes(2)
    expect(guided.reviewHistory.value).toEqual(['clue-1'])
    expect(guided.canReviewBack.value).toBe(true)
    expect(guided.phase.value).toBe('candidates')
    expect(guided.currentReviewClue.value).toBeNull()
  })

  it('returns to the review box when a blocking question appears during the candidate phase', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const reviewed = {
      id: 'reviewed-clue',
      dimension: 'language',
      value: 'ja',
      description: '已核对观察',
      source: 'model',
      confidence: 0.8,
      status: 'confirmed' as const,
    }
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({ revision: 2, clues: [reviewed] }),
      [{ target: japan, score: 0.8 }],
    ))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'pull-back' })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)
    expect(guided.candidatesVisible.value).toBe(true)

    api.dialogue.mockResolvedValueOnce(snapshot(
      dialogue({
        revision: 3,
        clues: [reviewed],
        pending_question: {
          id: 'ask-new',
          kind: 'ask',
          message: '补充确认',
          options: [{ value: 'yes', label: '是' }],
        },
      }),
      [{ target: japan, score: 0.8 }],
    ))
    await guided.reload()

    expect(guided.phase.value).toBe('review')
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
    expect(guided.locationConfirmationReady.value).toBe(false)
    expect(await guided.confirmLocation(japan)).toBe(false)
    expect(api.dialogueEvent).not.toHaveBeenCalled()
  })

  it('returns to review when a newly appended observation arrives during the candidate phase', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const reviewed = {
      id: 'reviewed-clue',
      dimension: 'language',
      value: 'ja',
      description: '已核对观察',
      source: 'model',
      confidence: 0.8,
      status: 'confirmed' as const,
    }
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({ revision: 2, clues: [reviewed] }),
      [{ target: japan, score: 0.8 }],
    ))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'new-clue' })

    await guided.start()
    expect(guided.nextPhase()).toBe(true)

    api.dialogue.mockResolvedValueOnce(snapshot(
      dialogue({
        revision: 3,
        clues: [
          reviewed,
          {
            id: 'new-clue',
            dimension: 'signage',
            value: '出口',
            description: '新发现观察',
            source: 'ocr',
            confidence: 0.7,
            status: 'pending',
          },
        ],
        pending_question: {
          id: 'review-new',
          kind: 'review_clue',
          clue_id: 'new-clue',
          message: '这条观察准确吗？',
        },
      }),
      [{ target: japan, score: 0.8 }],
    ))
    await guided.reload()

    expect(guided.phase.value).toBe('review')
    expect(guided.recentClueIds.value).toEqual(['new-clue'])
    expect(guided.currentReviewClue.value?.id).toBe('new-clue')
    expect(guided.reviewProgress.value).toEqual({ current: 2, total: 2 })
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
  })

  it('backtracks one level and shows the previous candidates only when review is ready', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    const reviewed = {
      id: 'reviewed-clue',
      dimension: 'language',
      value: 'ja',
      description: '已核对观察',
      source: 'model',
      confidence: 0.8,
      status: 'confirmed' as const,
    }
    api.dialogueStart.mockResolvedValue(snapshot(
      dialogue({
        revision: 5,
        stage: 'city',
        confirmed: { country: japan, admin1: hokkaido, city: null },
        clues: [reviewed],
      }),
      [{ target: sapporo, score: 0.7 }],
    ))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'back' })

    await guided.start()
    expect(guided.previousLevel.value).toBe('admin1')
    expect(guided.nextPhase()).toBe(true)

    api.dialogueEvent.mockResolvedValueOnce(snapshot(
      dialogue({
        revision: 6,
        stage: 'admin1',
        confirmed: { country: japan, admin1: null, city: null },
        clues: [reviewed],
      }),
      [{ target: hokkaido, score: 0.66 }],
    ))
    expect(await guided.backtrackPrevious()).toBe(true)
    expect(api.dialogueEvent.mock.calls[0]?.[1]).toMatchObject({ type: 'backtrack', level: 'admin1' })
    expect(guided.phase.value).toBe('candidates')
    expect(guided.candidatesVisible.value).toBe(true)
    expect(guided.topCandidates.value.map((item) => item.target.id)).toEqual(['JP.01'])
    expect(guided.previousLevel.value).toBe('country')
    expect(guided.dialogue.value?.confirmed.country?.id).toBe('JP')

    api.dialogueEvent.mockResolvedValueOnce(snapshot(
      dialogue({
        revision: 7,
        stage: 'country',
        clues: [{ ...reviewed, status: 'pending' }],
        pending_question: {
          id: 'review-back',
          kind: 'review_clue',
          clue_id: 'reviewed-clue',
          message: '这条观察准确吗？',
        },
      }),
      [{ target: japan, score: 0.9 }],
    ))
    expect(await guided.backtrackPrevious()).toBe(true)
    expect(guided.phase.value).toBe('review')
    expect(guided.candidatesVisible.value).toBe(false)
    expect(guided.topCandidates.value).toEqual([])
  })

  it('clears the phase and review history when the session switches', async () => {
    const sessionId = ref<number | null>(7)
    const api = apiMock()
    api.dialogueStart.mockResolvedValue(snapshot(dialogue({
      revision: 2,
      clues: [
        {
          id: 'clue-1',
          dimension: 'language',
          value: 'ja',
          description: '日语招牌',
          source: 'model',
          confidence: 0.8,
          status: 'pending',
        },
      ],
      pending_question: {
        id: 'review-1',
        kind: 'review_clue',
        clue_id: 'clue-1',
        message: '这条观察准确吗？',
      },
    })))
    api.dialogueEvent.mockResolvedValue(snapshot(dialogue({
      revision: 3,
      clues: [
        {
          id: 'clue-1',
          dimension: 'language',
          value: 'ja',
          description: '日语招牌',
          source: 'model',
          confidence: 0.8,
          status: 'confirmed',
        },
      ],
      pending_question: null,
    })))
    const guided = useGuidedDialogue(sessionId, { api, eventIdFactory: () => 'switch' })

    await guided.start()
    await guided.confirmClue('clue-1')
    expect(guided.nextPhase()).toBe(true)
    expect(guided.phase.value).toBe('candidates')
    expect(guided.reviewHistory.value).toEqual(['clue-1'])

    sessionId.value = 8

    expect(guided.phase.value).toBe('review')
    expect(guided.reviewHistory.value).toEqual([])
    expect(guided.canReviewBack.value).toBe(false)
    expect(guided.candidatesVisible.value).toBe(false)
  })

  it('keeps guided evidence separate from the legacy terminal country state', () => {
    const session = useGeoSession()
    session.applyUpload({
      session_id: 1,
      image_url: '/media/a.png',
      clues: [],
      scores: [],
      action: null,
      messages: [],
    })

    session.applyGuidedResponse(snapshot(dialogue({
      stage: 'resolved',
      confirmed: { country: japan, admin1: hokkaido, city: sapporo },
    })))

    expect(session.confirmedIso.value).toBeNull()
    expect(session.done.value).toBe(false)
  })
})




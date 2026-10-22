import { computed, ref, type Ref } from 'vue'
import type { useGuidedDialogue } from '@/composables/useGuidedDialogue'
import type { GuidedFreeTextSubmission } from '@/composables/useGuidedComposer'
import type {
  GuidedEventReceipt,
  ObservationAdoptionRequest,
  ObservationAdoptionStatus,
  ObservationAdoptionSubmission,
  RegionObservation,
  RegionReport,
} from '@/types'
import { guidedFreeTextReceiptOutcome } from '@/utils/guided'

type GuidedDialogueApi = ReturnType<typeof useGuidedDialogue>

/** Only the dialogue fields/actions consumed by guided submission state. */
export type GuidedSubmissionsApi = Pick<
  GuidedDialogueApi,
  'freeText' | 'adoptObservation' | 'retry' | 'isRetryableAdoption' | 'error' | 'conflict' | 'receipt'
>

export interface UseGuidedSubmissionsOptions {
  sessionId: Readonly<Ref<number | null>>
  sessionEpoch: Readonly<Ref<number>>
  busy: () => boolean
  regionReports: () => readonly RegionReport[]
  guided: GuidedSubmissionsApi
  guidedDraftText: Ref<string | null>
}

/** Free-text and OCR-adoption acknowledgements scoped to the current session. */
export function useGuidedSubmissions(options: UseGuidedSubmissionsOptions) {
  const guidedFreeTextSubmission = ref<GuidedFreeTextSubmission | null>(null)
  const observationAdoption = ref<ObservationAdoptionSubmission | null>(null)

  let guidedFreeTextNonce = 0
  let observationAdoptionNonce = 0

  const observationAdoptionRetryable = computed(() => {
    const adoption = observationAdoption.value
    return Boolean(
      adoption &&
        adoption.status === 'failed' &&
        options.guided.isRetryableAdoption(adoption.report_id, adoption.observation_id),
    )
  })

  function resetGuidedFreeTextSubmission(): void {
    guidedFreeTextNonce += 1
    guidedFreeTextSubmission.value = null
  }

  function resetObservationAdoption(): void {
    observationAdoptionNonce += 1
    observationAdoption.value = null
  }

  async function onGuidedFreeText(text: string): Promise<void> {
    const submitted = text.trim()
    if (!submitted) return

    const nonce = ++guidedFreeTextNonce
    const sessionId = options.sessionId.value
    guidedFreeTextSubmission.value = { text: submitted, status: 'pending', receipt: null }

    const applied = await options.guided.freeText(submitted)
    if (nonce !== guidedFreeTextNonce || options.sessionId.value !== sessionId) return

    if (!applied) {
      guidedFreeTextSubmission.value = {
        text: submitted,
        status: 'failed',
        error:
          options.guided.error.value ??
          options.guided.conflict.value ??
          '提交失败，请重试。',
        receipt: null,
      }
      return
    }

    // Snapshot the receipt produced by this exact event before any later guided
    // action can replace the shared latest receipt.
    const receipt = options.guided.receipt.value ? { ...options.guided.receipt.value } : null
    const outcome = guidedFreeTextReceiptOutcome(receipt)
    if (outcome.status === 'failed') {
      guidedFreeTextSubmission.value = {
        text: submitted,
        status: 'failed',
        error: outcome.error ?? '没有识别到新的线索，请修改后重试。',
        receipt,
      }
      return
    }

    options.guidedDraftText.value = null
    // Keep the server's distinction between an applied mutation and an
    // acknowledged pre-existing clue; never relabel unchanged as applied.
    guidedFreeTextSubmission.value = { text: submitted, status: outcome.status, receipt }
  }

  function findRegionObservation(
    reportId: string,
    observationId: string,
  ): { report: RegionReport; observation: RegionObservation } | null {
    const report = options.regionReports().find((item) => item.id === reportId)
    const observation = report?.observations.find((item) => item.id === observationId)
    return report && observation ? { report, observation } : null
  }

  function canAdoptRegionObservation(
    report: RegionReport,
    observation: RegionObservation,
  ): boolean {
    return (
      observation.kind === 'ocr_text' &&
      report.ocr.available &&
      report.ocr.state.toLowerCase() !== 'mock' &&
      Boolean(observation.text?.trim())
    )
  }

  function observationAdoptionOutcome(
    receipt: GuidedEventReceipt | null,
  ): { status: Exclude<ObservationAdoptionStatus, 'pending'>; error?: string | null } {
    const status = receipt?.status?.trim().toLowerCase() ?? ''
    if (status === 'applied') return { status: 'applied' }
    if (status === 'unchanged') return { status: 'unchanged' }
    if (status === 'unrecognized') {
      return {
        status: 'failed',
        error: receipt?.summary?.trim() || '无法确认这段文字，请重试。',
      }
    }
    return {
      status: 'unconfirmed',
      error: '没有收到确认，请重新加载后再试。',
    }
  }

  function completeObservationAdoption(
    base: ObservationAdoptionSubmission,
    applied: boolean,
  ): void {
    if (!applied) {
      observationAdoption.value = {
        ...base,
        status: 'failed',
        receipt: null,
        error:
          options.guided.error.value ??
          options.guided.conflict.value ??
          '确认文字失败，请重试。',
      }
      return
    }

    const receipt = options.guided.receipt.value ? { ...options.guided.receipt.value } : null
    const outcome = observationAdoptionOutcome(receipt)
    observationAdoption.value = {
      ...base,
      status: outcome.status,
      receipt,
      error: outcome.error ?? null,
    }
  }

  async function onAdoptObservation(request: ObservationAdoptionRequest): Promise<void> {
    if (options.busy()) return

    const reportId = request.report_id.trim()
    const observationId = request.observation_id.trim()
    if (!reportId || !observationId) return

    const match = findRegionObservation(reportId, observationId)
    if (!match || !canAdoptRegionObservation(match.report, match.observation)) {
      observationAdoption.value = {
        report_id: reportId,
        observation_id: observationId,
        text: match?.observation.text?.trim() ?? '',
        status: 'failed',
        receipt: null,
        error: '这段文字现在不能使用，请重新分析区域。',
      }
      return
    }

    const sessionId = options.sessionId.value
    if (sessionId === null) return
    const epoch = options.sessionEpoch.value
    const nonce = ++observationAdoptionNonce
    const base: ObservationAdoptionSubmission = {
      report_id: reportId,
      observation_id: observationId,
      text: match.observation.text!.trim(),
      status: 'pending',
      receipt: null,
      error: null,
    }
    observationAdoption.value = base

    const applied = await options.guided.adoptObservation(reportId, observationId)
    if (
      nonce !== observationAdoptionNonce ||
      options.sessionId.value !== sessionId ||
      epoch !== options.sessionEpoch.value
    ) {
      return
    }
    completeObservationAdoption(base, applied)
  }

  async function retryObservationAdoption(): Promise<void> {
    if (options.busy()) return

    const current = observationAdoption.value
    if (
      !current ||
      current.status !== 'failed' ||
      !options.guided.isRetryableAdoption(current.report_id, current.observation_id)
    ) {
      return
    }

    const sessionId = options.sessionId.value
    if (sessionId === null) return
    const epoch = options.sessionEpoch.value
    const nonce = ++observationAdoptionNonce
    const base: ObservationAdoptionSubmission = {
      ...current,
      status: 'pending',
      receipt: null,
      error: null,
    }
    observationAdoption.value = base

    const applied = await options.guided.retry()
    if (
      nonce !== observationAdoptionNonce ||
      options.sessionId.value !== sessionId ||
      epoch !== options.sessionEpoch.value
    ) {
      return
    }
    completeObservationAdoption(base, applied)
  }

  async function retryGuidedOperation(): Promise<void> {
    const current = observationAdoption.value
    if (
      current?.status === 'failed' &&
      options.guided.isRetryableAdoption(current.report_id, current.observation_id)
    ) {
      await retryObservationAdoption()
      return
    }
    await options.guided.retry()
  }

  return {
    guidedFreeTextSubmission,
    observationAdoption,
    observationAdoptionRetryable,
    resetGuidedFreeTextSubmission,
    resetObservationAdoption,
    onGuidedFreeText,
    onAdoptObservation,
    retryObservationAdoption,
    retryGuidedOperation,
  }
}

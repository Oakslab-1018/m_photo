import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import {
  useGuidedSubmissions,
  type GuidedSubmissionsApi,
} from '@/composables/useGuidedSubmissions'
import type {
  GuidedEventReceipt,
  ObservationAdoptionRequest,
  RegionObservation,
  RegionReport,
} from '@/types'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

interface ReportOptions {
  reportId?: string
  observationId?: string
  ocrAvailable?: boolean
  ocrState?: string
  kind?: RegionObservation['kind']
  text?: string | null
}

function ocrReport(options: ReportOptions = {}): RegionReport {
  const reportId = options.reportId ?? 'report-1'
  const observationId = options.observationId ?? 'observation-1'
  const text = options.text === undefined ? '  汉口路  ' : options.text
  return {
    id: reportId,
    purpose: 'ocr',
    scale: 1,
    bbox_px: [0, 0, 100, 50],
    bbox_norm: { x: 0, y: 0, w: 1, h: 1 },
    image: {
      width: 100,
      height: 100,
      orientation_applied: false,
      detail_width: 100,
      detail_height: 100,
      ocr_scale: 1,
    },
    quality: { issues: [], metrics: {} },
    ocr: {
      provider: 'test',
      enabled: true,
      available: options.ocrAvailable ?? true,
      state: options.ocrState ?? 'ready',
      line_count: text ? 1 : 0,
    },
    model: {
      provider: 'test',
      enabled: false,
      available: false,
      state: 'disabled',
      ranking_applied: false,
    },
    observations: [
      {
        id: observationId,
        kind: options.kind ?? 'ocr_text',
        dimension: 'signage',
        value: text?.trim() ?? '',
        text,
        label: '招牌',
        evidence: 'OCR',
        source: 'ocr',
        trust: {
          level: 'high',
          reliable: true,
          validated: true,
          affects_ranking: true,
        },
      },
    ],
    question_suggestions: [],
    summary: '区域 OCR',
    dedupe: 'new',
  }
}

function setup(initialReports: RegionReport[] = [ocrReport()]) {
  const sessionId = ref<number | null>(7)
  const sessionEpoch = ref(0)
  const busy = ref(false)
  const guidedDraftText = ref<string | null>(null)
  const reports = ref<RegionReport[]>(initialReports)
  const error = ref<string | null>(null)
  const conflict = ref<string | null>(null)
  const receipt = ref<GuidedEventReceipt | null>(null)

  let retryableAdoption: ObservationAdoptionRequest | null = null
  const freeText = vi.fn<(text: string) => Promise<boolean>>().mockResolvedValue(true)
  const adoptObservation = vi
    .fn<(reportId: string, observationId: string) => Promise<boolean>>()
    .mockResolvedValue(true)
  const retry = vi.fn<() => Promise<boolean>>().mockResolvedValue(true)
  const isRetryableAdoption = vi.fn(
    (reportId: string, observationId: string) =>
      retryableAdoption?.report_id === reportId &&
      retryableAdoption.observation_id === observationId,
  )

  const guided: GuidedSubmissionsApi = {
    freeText,
    adoptObservation,
    retry,
    isRetryableAdoption,
    error,
    conflict,
    receipt,
  }

  const submissions = useGuidedSubmissions({
    sessionId,
    sessionEpoch,
    busy: () => busy.value,
    regionReports: () => reports.value,
    guided,
    guidedDraftText,
  })

  return {
    sessionId,
    sessionEpoch,
    busy,
    guidedDraftText,
    reports,
    error,
    conflict,
    receipt,
    freeText,
    adoptObservation,
    retry,
    isRetryableAdoption,
    setRetryableAdoption: (request: ObservationAdoptionRequest | null) => {
      retryableAdoption = request
    },
    submissions,
  }
}

const adoptionRequest: ObservationAdoptionRequest = {
  report_id: 'report-1',
  observation_id: 'observation-1',
}

describe('guided submissions', () => {
  it('drops a free-text completion after its submission is reset', async () => {
    const context = setup()
    const pending = deferred<boolean>()
    context.freeText.mockReturnValueOnce(pending.promise)
    context.guidedDraftText.value = '正在编辑的新线索'

    const operation = context.submissions.onGuidedFreeText('  汉口路  ')
    expect(context.freeText).toHaveBeenCalledWith('汉口路')
    expect(context.submissions.guidedFreeTextSubmission.value).toMatchObject({
      text: '汉口路',
      status: 'pending',
    })

    context.submissions.resetGuidedFreeTextSubmission()
    context.receipt.value = { event_id: 'stale', status: 'applied' }
    pending.resolve(true)
    await operation

    expect(context.submissions.guidedFreeTextSubmission.value).toBeNull()
    expect(context.guidedDraftText.value).toBe('正在编辑的新线索')
  })

  it('drops an observation adoption completion after the session switches', async () => {
    const context = setup()
    const pending = deferred<boolean>()
    context.adoptObservation.mockReturnValueOnce(pending.promise)

    const operation = context.submissions.onAdoptObservation(adoptionRequest)
    expect(context.submissions.observationAdoption.value).toMatchObject({
      report_id: 'report-1',
      observation_id: 'observation-1',
      text: '汉口路',
      status: 'pending',
    })

    context.sessionId.value = 8
    context.receipt.value = { event_id: 'old-session', status: 'applied' }
    pending.resolve(true)
    await operation

    // The session guard, rather than the reset nonce, prevents the stale write.
    expect(context.submissions.observationAdoption.value?.status).toBe('pending')
    expect(context.submissions.observationAdoption.value?.receipt).toBeNull()
  })

  it('snapshots the exact free-text receipt and preserves unchanged status', async () => {
    const context = setup()
    const exactReceipt: GuidedEventReceipt = {
      event_id: 'free-text-1',
      status: 'unchanged',
      summary: '该线索此前已保存。',
    }
    context.freeText.mockImplementationOnce(async () => {
      context.receipt.value = exactReceipt
      return true
    })

    await context.submissions.onGuidedFreeText(' 玉林路 ')
    const submission = context.submissions.guidedFreeTextSubmission.value

    expect(submission).toMatchObject({
      text: '玉林路',
      status: 'unchanged',
      receipt: exactReceipt,
    })
    expect(submission?.receipt).not.toBe(exactReceipt)
    expect(context.guidedDraftText.value).toBeNull()

    context.receipt.value = { event_id: 'later-event', status: 'applied' }
    exactReceipt.status = 'unrecognized'
    expect(submission?.receipt).toEqual({
      event_id: 'free-text-1',
      status: 'unchanged',
      summary: '该线索此前已保存。',
    })
  })

  it.each([
    {
      label: 'applied',
      receipt: { event_id: 'adopt-applied', status: 'applied' } as GuidedEventReceipt,
      status: 'applied',
      error: null,
    },
    {
      label: 'unchanged',
      receipt: { event_id: 'adopt-unchanged', status: 'unchanged' } as GuidedEventReceipt,
      status: 'unchanged',
      error: null,
    },
    {
      label: 'unrecognized',
      receipt: {
        event_id: 'adopt-unrecognized',
        status: 'unrecognized',
        summary: '  这段文字无法确认。  ',
      } as GuidedEventReceipt,
      status: 'failed',
      error: '这段文字无法确认。',
    },
    {
      label: 'unconfirmed',
      receipt: null,
      status: 'unconfirmed',
      error: '没有收到确认，请重新加载后再试。',
    },
  ])('maps an $label adoption receipt without upgrading its status', async (caseData) => {
    const context = setup()
    context.adoptObservation.mockImplementationOnce(async () => {
      context.receipt.value = caseData.receipt
      return true
    })

    await context.submissions.onAdoptObservation(adoptionRequest)
    const adoption = context.submissions.observationAdoption.value

    expect(adoption?.status).toBe(caseData.status)
    expect(adoption?.error ?? null).toBe(caseData.error)
    if (caseData.receipt) {
      expect(adoption?.receipt).toEqual(caseData.receipt)
      expect(adoption?.receipt).not.toBe(caseData.receipt)
    } else {
      expect(adoption?.receipt).toBeNull()
    }
  })

  it.each([
    {
      label: 'unavailable OCR',
      report: ocrReport({ ocrAvailable: false }),
    },
    {
      label: 'mock OCR',
      report: ocrReport({ ocrState: 'MOCK' }),
    },
    {
      label: 'non-OCR observation',
      report: ocrReport({ kind: 'semantic' }),
    },
  ])('blocks $label adoption without calling the guided API', async ({ report }) => {
    const context = setup([report])

    await context.submissions.onAdoptObservation(adoptionRequest)

    expect(context.adoptObservation).not.toHaveBeenCalled()
    expect(context.submissions.observationAdoption.value).toMatchObject({
      report_id: 'report-1',
      observation_id: 'observation-1',
      status: 'failed',
      receipt: null,
      error: '这段文字现在不能使用，请重新分析区域。',
    })
  })

  it('routes a retryable failed adoption through the guarded adoption retry', async () => {
    const context = setup()
    context.error.value = '网络中断'
    context.adoptObservation.mockResolvedValueOnce(false)

    await context.submissions.onAdoptObservation(adoptionRequest)
    expect(context.submissions.observationAdoption.value).toMatchObject({
      status: 'failed',
      error: '网络中断',
      receipt: null,
    })

    context.setRetryableAdoption(adoptionRequest)
    expect(context.submissions.observationAdoptionRetryable.value).toBe(true)

    // Routing through retryObservationAdoption honors the shared busy guard.
    context.busy.value = true
    await context.submissions.retryGuidedOperation()
    expect(context.retry).not.toHaveBeenCalled()

    context.busy.value = false
    const pendingRetry = deferred<boolean>()
    context.retry.mockReturnValueOnce(pendingRetry.promise)
    const operation = context.submissions.retryGuidedOperation()
    expect(context.submissions.observationAdoption.value).toMatchObject({
      status: 'pending',
      receipt: null,
      error: null,
      text: '汉口路',
    })

    context.receipt.value = { event_id: 'adopt-retry', status: 'applied' }
    pendingRetry.resolve(true)
    await operation

    expect(context.retry).toHaveBeenCalledTimes(1)
    expect(context.adoptObservation).toHaveBeenCalledTimes(1)
    expect(context.submissions.observationAdoption.value?.status).toBe('applied')
    expect(context.submissions.observationAdoptionRetryable.value).toBe(false)
  })

  it('falls back to the guided retry when no retryable adoption is pending', async () => {
    const context = setup()

    await context.submissions.retryGuidedOperation()

    expect(context.retry).toHaveBeenCalledTimes(1)
  })
})


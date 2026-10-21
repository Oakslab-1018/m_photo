<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '@/api'
import { normalizeApiError } from '@/api/errors'
import type {
  AnalyzePointRequest,
  GuidedDialogueResponse,
  Country,
  CountryScore,
  Clue,
  FeedbackPayload,
  FeedbackResponse,
  LocationTarget,
  PendingChatMessage,
  RegionAnalysisRecord,
  RegionQuestion,
  RegionReport,
} from '@/types'
import { useGeoSession } from '@/composables/useGeoSession'
import { useGuidedDialogue } from '@/composables/useGuidedDialogue'
import { useImmersiveLayout } from '@/composables/useImmersiveLayout'
import type { GuidedFreeTextSubmission } from '@/composables/useGuidedComposer'
import { clueIdentity, clueValueLabel, countGroupedClues, dimensionLabel } from '@/utils/clues'
import { isQuestionAction } from '@/utils/dialogue'
import {
  guidedFreeTextReceiptOutcome,
  guidedQuestionMessage,
  guidedStageTitle,
  guidedTargetKey,
  pendingQuestionClue,
  safeGuidedText,
  targetLabel,
} from '@/utils/guided'
import { feedbackReceipt, receiptText, type FeedbackReceiptTone } from '@/utils/feedback'
import {
  mergeRegionReports,
  newRegionRequestId,
  normalizeAnalyzePointRequest,
  normalizeRegionReport,
  regionReportsFromMessages,
} from '@/utils/regionAnalysis'
import {
  CHINA_REGION_LABELS,
  catalogCountryLabel,
  displayCountryName,
  entryCode,
  exclusionLabel,
  isChinaRegionCode,
  normalizeCountryIso,
  normalizeIsoCode,
  regionScopeLabel,
  type SourceRegionEntry,
} from '@/utils/geo'
import UploadPanel from '@/components/UploadPanel.vue'
import ClueList from '@/components/ClueList.vue'
import ChatLog from '@/components/ChatLog.vue'
import GeoMap from '@/components/GeoMap.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import CandidatePanel from '@/components/CandidatePanel.vue'
import ConfirmLocationDialog from '@/components/ConfirmLocationDialog.vue'
import GuidedDialoguePanel from '@/components/GuidedDialoguePanel.vue'
import ImageViewer from '@/components/ImageViewer.vue'

const session = useGeoSession()
const guided = useGuidedDialogue(session.sessionId, {
  onSnapshot: applyGuidedSnapshot,
})

function applyGuidedSnapshot(response: GuidedDialogueResponse): void {
  session.applyGuidedResponse(response)
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024

const selectedFile = ref<File | null>(null)
const localPreviewUrl = ref<string | null>(null)
const uploading = ref(false)
const feedbackBusy = ref(false)
const pointBusy = ref(false)
const cityBusy = ref(false)
const errorMessage = ref<string | null>(null)
const uploadNotice = ref<string | null>(null)
const cityNotice = ref<string | null>(null)
const pointStatus = ref('点选或框选区域后，点击“分析选定区域”；分析结果只作为线索。')
const pointStatusKind = ref<'idle' | 'loading' | 'success' | 'error'>('idle')
const viewerOpen = ref(false)
const focusedClue = ref<Clue | null>(null)
const clickPoints = ref<Array<{ x: number; y: number }>>([])
const confirmCandidate = ref<(typeof session.scores.value)[number] | null>(null)
const selectedIso = ref<string | null>(null)
const selectedGuidedTargetKey = ref<string | null>(null)
const countryDirectory = ref<Country[]>([])
const feedbackDraft = ref('')
const pendingMessages = ref<PendingChatMessage[]>([])
const lastOutcome = ref<{ tone: FeedbackReceiptTone | 'info'; text: string } | null>(null)
const citySourceIso = ref<string | null>(null)
const citySourceLabel = ref<string | null>(null)
const knownCountryNames = ref<Record<string, string>>({})
const regionRequests = ref<RegionAnalysisRecord[]>([])
const immediateRegionReports = ref<RegionReport[]>([])
const selectedRegionReportId = ref<string | null>(null)
const guidedDraftText = ref<string | null>(null)
const guidedFreeTextSubmission = ref<GuidedFreeTextSubmission | null>(null)

type FocusMode = 'collab' | 'map' | 'photo'
type MobileSheetState = 'peek' | 'half' | 'full'
type EvidenceTab = 'photo' | 'clues' | 'references'

const FOCUS_MODES: Array<{ key: FocusMode; label: string; hint: string }> = [
  { key: 'collab', label: '协同推演', hint: '同时展开证据与推理' },
  { key: 'map', label: '地图专注', hint: '收起两侧浮层，保留完整地图' },
  { key: 'photo', label: '照片取证', hint: '展开证据台并收起推理' },
]

const commandBarRef = ref<HTMLElement | null>(null)
const mapHostRef = ref<HTMLElement | null>(null)
const evidenceDockRef = ref<HTMLElement | null>(null)
const reasoningDockRef = ref<HTMLElement | null>(null)
const focusMode = ref<FocusMode>('collab')
const leftCollapsed = ref(false)
const rightCollapsed = ref(false)
const reasoningWide = ref(false)
const mobileSheetState = ref<MobileSheetState>('half')
const mobileEvidenceOpen = ref(false)
const mobileEvidenceTab = ref<EvidenceTab>('photo')
const historyOpen = ref(false)
const historyDialogRef = ref<HTMLElement | null>(null)
const mobileEvidenceDialogRef = ref<HTMLElement | null>(null)

const { mapInsets, isMobile, isCompact } = useImmersiveLayout({
  commandBar: commandBarRef,
  mapHost: mapHostRef,
  leftDock: evidenceDockRef,
  rightDock: reasoningDockRef,
  includeTop: () => true,
  includeLeft: () => focusMode.value !== 'map',
  includeRight: () => focusMode.value !== 'map',
  includeBottom: () => !mobileEvidenceOpen.value,
})

/** Monotonic session generation: stale responses can never overwrite newer state. */
const sessionEpoch = ref(0)
/** Monotonic city-request nonce: older city responses must not replace newer country picks. */
let cityRequestNonce = 0
let pendingNonce = 0
let regionRequestNonce = 0
let guidedFreeTextNonce = 0

const regionReports = computed(() =>
  mergeRegionReports(regionReportsFromMessages(session.messages.value), immediateRegionReports.value),
)

const busy = computed(
  () =>
    uploading.value ||
    feedbackBusy.value ||
    pointBusy.value ||
    cityBusy.value ||
    guided.loading.value ||
    guided.applying.value,
)
const hasSession = computed(() => session.sessionId.value !== null)
const previewUrl = computed(() => localPreviewUrl.value ?? session.imageUrl.value)
/** Card count after grouping/dedupe, matching what the clue panel actually renders. */
const clueCount = computed(() => countGroupedClues(session.clues.value))
const recentClueKeys = computed(() => {
  const recent = new Set(guided.recentClueIds.value)
  return (guided.dialogue.value?.clues ?? [])
    .filter((clue) => recent.has(clue.id))
    .map((clue) => clueIdentity(clue.dimension, clue.value))
})
const questionPending = computed(() => {
  if (!guided.active.value) return isQuestionAction(session.action.value)
  const question = guided.pendingQuestion.value
  if (!question) return false
  // A matched country confidence route presents routine review as non-blocking.
  if (guided.confidenceShortcut.value) return false
  // The default confirm_location prompt is not an unmet review item: it is
  // answered by the explicit candidate step, never by an answer chip.
  return question.kind !== 'confirm_location'
})
const confirmSubregion = computed(() =>
  confirmCandidate.value ? regionScopeLabel(confirmCandidate.value) : null,
)

/** Region-aware exclusion chips: raw TW/HK/MO never render as 中国 excluded. */
const exclusionChips = computed(() =>
  session.exclusions.value.map((iso) => ({ iso: normalizeIsoCode(iso), label: exclusionLabel(iso) })),
)

const guidedRankingsUnlocked = computed(() => guided.rankingsUnlocked.value)
/** Unified candidate gate: hidden until the explicit 下一步 (or back navigation). */
const guidedCandidatesVisible = computed(() => guided.candidatesVisible.value)
const guidedConfirmationReady = computed(() => guided.locationConfirmationReady.value)
const guidedDataReady = computed(() => guided.dialogue.value !== null)
const guidedVisibleCandidates = computed(() =>
  guided.confidenceShortcut.value
    ? guided.confidenceCandidates.value
    : guided.topCandidates.value,
)
const guidedCandidateSurfaceVisible = computed(
  () => guidedCandidatesVisible.value || guided.confidenceShortcut.value !== null,
)
const guidedMapStage = computed(() => {
  if (!guided.active.value || !guided.dialogue.value) return null
  if (guided.dialogue.value.stage === 'resolved') return 'resolved'
  return guidedCandidateSurfaceVisible.value ? guided.dialogue.value.stage : null
})
const guidedPendingQuestionText = computed(() => {
  const dialogue = guided.dialogue.value
  const question = guided.pendingQuestion.value
  if (!dialogue || !question) return ''
  return guidedQuestionMessage(
    question,
    pendingQuestionClue(dialogue),
    guidedRankingsUnlocked.value,
  )
})
/** Never fall back to legacy score rows while guided data is loading or locked. */
const guidedCountryScores = computed(() => {
  if (!guided.active.value) return session.scores.value
  if (!guidedDataReady.value || !guidedCandidateSurfaceVisible.value) return []
  const visible = new Set(
    guidedVisibleCandidates.value
      .filter((candidate) => candidate.target.level === 'country')
      .map((candidate) => normalizeIsoCode(candidate.target.id)),
  )
  return session.scores.value.filter((score) => visible.has(normalizeIsoCode(score.iso)))
})
const visibleCityScores = computed(() =>
  guided.active.value && !guidedCandidateSurfaceVisible.value ? [] : session.cityScores.value,
)

const countryOptions = computed(() => {
  const directory = countryDirectory.value
  if (directory.length) {
    const seen = new Set<string>()
    const options: Array<{ iso: string; name_zh: string }> = []
    for (const country of directory) {
      const iso = normalizeIsoCode(country.iso)
      // TW/HK/MO stay addressable through the CN region-scope selector only.
      if (!iso || isChinaRegionCode(iso) || seen.has(iso)) continue
      seen.add(iso)
      options.push({ iso, name_zh: catalogCountryLabel(country) })
    }
    options.sort((a, b) => a.name_zh.localeCompare(b.name_zh, 'zh-Hans-CN'))
    return options
  }
  return session.scores.value.map((score) => ({ iso: score.iso, name_zh: displayCountryName(score) }))
})

/** Degraded-inference explanation; always visible, never inside a collapsed block. */
const inferenceNotice = computed(() => safeGuidedText(session.inferenceNotice.value) || null)

const sessionStatus = computed(() => {
  if (uploading.value) return '正在分析图像…'
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return guided.loading.value ? '正在启动引导定位…' : '引导定位待重试'
    if (current.stage === 'resolved') {
      return `已完成：${targetLabel(current.confirmed.city)}`
    }
    if (!guidedCandidateSurfaceVisible.value) {
      if (guided.reviewReady.value) return '观察已核对 · 点击“下一步”查看候选'
      const review = guided.reviewProgress.value
      return review
        ? `先核对观察 · 第 ${review.current} / ${review.total}`
        : `先核对观察 · 待处理 ${guided.pendingClueCount.value} 条`
    }
    if (!guidedConfirmationReady.value) {
      return `${guidedStageTitle(current.stage)} · 候选已解锁 · 新增观察待核对`
    }
    return `${guidedStageTitle(current.stage)} · 候选 ${guidedVisibleCandidates.value.length} 个`
  }
  if (session.confirmedIso.value) {
    const country = session.scores.value.find((item) => item.iso === session.confirmedIso.value)
    const scope = country ? regionScopeLabel(country) : null
    const name = country ? displayCountryName(country) : session.confirmedIso.value
    return `已由用户确认：${name}${scope ? `（区域命中 ${scope}）` : ''}`
  }
  if (!hasSession.value) return '未开始 · 请上传一张照片'
  return `候选国家 ${session.scores.value.length} 个 · 均为代表点`
})

const countryBreadcrumb = computed(() => {
  const target = guided.dialogue.value?.confirmed.country
  if (target) return targetLabel(target)
  if (session.confirmedIso.value) return countryLabel(session.confirmedIso.value)
  if (guided.active.value) {
    return guidedCandidateSurfaceVisible.value ? '国家待确认' : '国家 / 地区'
  }
  return hasSession.value ? '国家待确认' : '国家待定'
})

const cityBreadcrumb = computed(() => {
  const target = guided.dialogue.value?.confirmed.city
  if (target) return targetLabel(target)
  if (guided.active.value && guided.dialogue.value?.stage === 'city') return '城市待确认'
  if (guided.active.value && guided.dialogue.value?.stage === 'resolved') return '城市已记录'
  if (!guided.active.value && session.cityScores.value.length) return '城市诊断'
  return '城市待定'
})

const historyAllowed = computed(
  () => !guided.active.value || guidedCandidateSurfaceVisible.value,
)

const mobileEvidenceModalOpen = computed(
  () => isMobile.value && mobileEvidenceOpen.value,
)
const backgroundInert = computed(
  () => historyOpen.value || mobileEvidenceModalOpen.value,
)
const activeModal = computed<'history' | 'evidence' | null>(() => {
  if (historyOpen.value) return 'history'
  if (mobileEvidenceModalOpen.value) return 'evidence'
  return null
})

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[contenteditable="true"]',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

let modalReturnFocus: HTMLElement | null = null
let skipModalFocusRestore = false
let modalFocusEpoch = 0

function modalElement(modal: 'history' | 'evidence'): HTMLElement | null {
  return modal === 'history' ? historyDialogRef.value : mobileEvidenceDialogRef.value
}

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) => element.getClientRects().length > 0 && element.getAttribute('aria-hidden') !== 'true',
  )
}

function focusActiveModal(modal: 'history' | 'evidence'): void {
  const container = modalElement(modal)
  if (!container) return
  const preferred = container.querySelector<HTMLElement>('[data-modal-autofocus]')
  const target = preferred ?? focusableElements(container)[0] ?? container
  target.focus()
}

function trapModalFocus(event: KeyboardEvent, modal: 'history' | 'evidence'): void {
  const container = modalElement(modal)
  if (!container) return
  const focusable = focusableElements(container)
  if (!focusable.length) {
    event.preventDefault()
    container.focus()
    return
  }
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  const active = document.activeElement
  if (event.shiftKey && (active === first || active === container || !container.contains(active))) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && (active === last || active === container || !container.contains(active))) {
    event.preventDefault()
    first.focus()
  }
}

function restoreModalFocus(target: HTMLElement | null): void {
  if (!target?.isConnected || target.getClientRects().length === 0) return
  target.focus()
}
const evidenceAnchorVisible = computed(
  () => (isMobile.value ? mobileEvidenceOpen.value : focusMode.value !== 'map'),
)
const reasoningAnchorVisible = computed(
  () => (isMobile.value ? !mobileEvidenceOpen.value : focusMode.value !== 'map'),
)
const showEvidenceThumbnail = computed(() => !isMobile.value && leftCollapsed.value)
const showReasoningTaskChip = computed(() => !isMobile.value && rightCollapsed.value)
function setFocusMode(mode: FocusMode): void {
  focusMode.value = mode
  historyOpen.value = false
  if (isMobile.value) {
    if (mode === 'photo') {
      mobileEvidenceOpen.value = true
      mobileEvidenceTab.value = 'photo'
      mobileSheetState.value = 'peek'
    } else if (mode === 'map') {
      mobileEvidenceOpen.value = false
      mobileSheetState.value = 'peek'
    } else {
      mobileEvidenceOpen.value = false
      mobileSheetState.value = 'half'
    }
    return
  }
  if (mode === 'photo') {
    leftCollapsed.value = false
    rightCollapsed.value = true
  } else if (mode === 'map') {
    leftCollapsed.value = true
    rightCollapsed.value = true
  } else {
    rightCollapsed.value = false
    if (!isCompact.value) leftCollapsed.value = false
  }
}

function expandEvidenceDock(): void {
  leftCollapsed.value = false
  if (focusMode.value === 'map') focusMode.value = 'collab'
}

function collapseEvidenceDock(): void {
  leftCollapsed.value = true
}

function expandReasoningDock(): void {
  rightCollapsed.value = false
  if (focusMode.value === 'map') focusMode.value = 'collab'
}

function collapseReasoningDock(): void {
  rightCollapsed.value = true
}

function toggleReasoningWidth(): void {
  reasoningWide.value = !reasoningWide.value
}

function setMobileSheetState(state: MobileSheetState): void {
  mobileSheetState.value = state
}

function openMobileEvidence(tab: EvidenceTab): void {
  historyOpen.value = false
  mobileEvidenceTab.value = tab
  mobileEvidenceOpen.value = true
  mobileSheetState.value = 'peek'
}

function closeMobileEvidence(): void {
  mobileEvidenceOpen.value = false
}

function openHistory(): void {
  if (!historyAllowed.value) return
  mobileEvidenceOpen.value = false
  historyOpen.value = true
}

function closeHistory(): void {
  historyOpen.value = false
}

function openFocusedClue(clue: Clue): void {
  if (!session.imageUrl.value) return
  focusedClue.value = clue
  if (mobileEvidenceOpen.value) skipModalFocusRestore = true
  mobileEvidenceOpen.value = false
  viewerOpen.value = true
}

function openImageViewer(): void {
  if (!session.imageUrl.value) return
  focusedClue.value = null
  if (mobileEvidenceOpen.value) skipModalFocusRestore = true
  mobileEvidenceOpen.value = false
  viewerOpen.value = true
}

function onGlobalKeydown(event: KeyboardEvent): void {
  const modal = activeModal.value
  if (!modal) return
  if (event.key === 'Tab') {
    trapModalFocus(event, modal)
    return
  }
  if (event.key !== 'Escape') return
  event.preventDefault()
  event.stopPropagation()
  if (modal === 'history') closeHistory()
  else closeMobileEvidence()
}

const liveStatus = computed<{ tone: FeedbackReceiptTone | 'info'; text: string }>(() => {
  const failed = pendingMessages.value.find((item) => item.status === 'failed')
  if (failed) return { tone: 'warn', text: `发送失败：${failed.error ?? '未知错误'} · 可在对话中重试或删除` }
  if (guided.conflict.value) return { tone: 'warn', text: safeGuidedText(guided.conflict.value) }
  if (guided.error.value) return { tone: 'warn', text: safeGuidedText(guided.error.value) }
  if (
    (guided.applying.value || guided.loading.value) &&
    guided.pendingOperation.value
  ) {
    return { tone: 'info', text: guided.pendingOperation.value.label }
  }
  if (guided.applying.value) return { tone: 'info', text: '正在提交引导操作…' }
  if (guided.loading.value) return { tone: 'info', text: '正在同步引导状态…' }
  if (feedbackBusy.value) return { tone: 'info', text: '正在提交反馈，等待服务器回执…' }
  if (uploading.value) return { tone: 'info', text: '正在分析图像…' }
  if (lastOutcome.value) return lastOutcome.value
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return { tone: 'muted', text: '等待引导状态' }
    if (current.stage === 'resolved') return { tone: 'ok', text: `已确认城市：${targetLabel(current.confirmed.city)}` }
    if (!guidedCandidateSurfaceVisible.value) {
      if (guided.reviewReady.value) {
        return { tone: 'ok', text: '观察已核对 · 点击“下一步”查看候选' }
      }
      const review = guided.reviewProgress.value
      return {
        tone: 'info',
        text: guidedPendingQuestionText.value
          || (review
            ? `请先核对观察 · 第 ${review.current} / ${review.total}`
            : `请先核对观察 · 待处理 ${guided.pendingClueCount.value} 条`),
      }
    }
    if (!guidedConfirmationReady.value) {
      return { tone: 'muted', text: '新增观察待核对，暂不能确认地点' }
    }
    if (current.pending_question && !guided.confidenceShortcut.value) {
      return { tone: 'info', text: guidedPendingQuestionText.value || '请继续完成观察核对。' }
    }
    return { tone: 'muted', text: `${guidedStageTitle(current.stage)} · 请确认或排除候选` }
  }
  if (hasSession.value) {
    if (session.confirmedIso.value) {
      const country = session.scores.value.find((item) => item.iso === session.confirmedIso.value)
      const scope = country ? regionScopeLabel(country) : null
      return {
        tone: 'ok',
        text: `已确认：${country ? displayCountryName(country) : session.confirmedIso.value}${scope ? `（区域命中 ${scope}）` : ''}`,
      }
    }
    return { tone: 'muted', text: `候选 ${session.scores.value.length} 个 · 可确认或排除候选` }
  }
  return { tone: 'muted', text: '上传照片后开始人机协同推理' }
})

watch(
  isCompact,
  (compact) => {
    if (compact && !isMobile.value) leftCollapsed.value = true
  },
  { immediate: true },
)

watch(isMobile, (mobile) => {
  historyOpen.value = false
  mobileEvidenceOpen.value = false
  if (mobile) mobileSheetState.value = 'half'
})

watch(historyAllowed, (allowed) => {
  if (!allowed) historyOpen.value = false
})

watch(
  activeModal,
  async (next, previous) => {
    const epoch = ++modalFocusEpoch
    if (next) {
      if (!previous) {
        modalReturnFocus = document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null
      }
      await nextTick()
      if (epoch === modalFocusEpoch) focusActiveModal(next)
      return
    }
    if (!previous) return

    const restoreTarget = modalReturnFocus
    modalReturnFocus = null
    await nextTick()
    if (epoch !== modalFocusEpoch) return
    if (skipModalFocusRestore) {
      skipModalFocusRestore = false
      return
    }
    restoreModalFocus(restoreTarget)
  },
)

watch(
  [session.scores, countryDirectory],
  () => {
    const next = { ...knownCountryNames.value }
    for (const score of session.scores.value) next[normalizeCountryIso(score.iso)] = displayCountryName(score)
    for (const country of countryDirectory.value) {
      const iso = normalizeIsoCode(country.iso)
      if (iso) next[iso] = catalogCountryLabel(country)
    }
    if (!next.CN) next.CN = '中国'
    knownCountryNames.value = next
  },
  { immediate: true },
)

onMounted(async () => {
  try {
    countryDirectory.value = await api.countries()
  } catch {
    // Directory is a convenience only; fall back to the current top candidates.
    countryDirectory.value = []
  }
})

onMounted(() => {
  window.addEventListener('keydown', onGlobalKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onGlobalKeydown)
})

function setError(caught: unknown, fallback: string): void {
  errorMessage.value = normalizeApiError(caught, undefined, fallback).message
}

function releasePreviewUrl(): void {
  if (localPreviewUrl.value?.startsWith('blob:')) URL.revokeObjectURL(localPreviewUrl.value)
  localPreviewUrl.value = null
}

function resetGuidedFreeTextSubmission(): void {
  guidedFreeTextNonce += 1
  guidedFreeTextSubmission.value = null
}

function resetTransientUi(): void {
  viewerOpen.value = false
  focusedClue.value = null
  historyOpen.value = false
  mobileEvidenceOpen.value = false
  confirmCandidate.value = null
  clickPoints.value = []
  selectedIso.value = null
  selectedGuidedTargetKey.value = null
  cityNotice.value = null
  citySourceIso.value = null
  citySourceLabel.value = null
  pointStatus.value = '点选或框选区域后，点击“分析选定区域”；分析结果只作为线索。'
  pointStatusKind.value = 'idle'
  resetGuidedFreeTextSubmission()
  resetRegionAnalysisState()
}

function cancelInFlightUi(): void {
  uploading.value = false
  feedbackBusy.value = false
  pointBusy.value = false
  cityBusy.value = false
}

/**
 * The session/epoch these bubbles belonged to is gone. Keep them visible and
 * dismissible, but block retry: an old photo's feedback must never be silently
 * sent against the new session.
 */
function orphanPending(reason: string): void {
  pendingMessages.value = pendingMessages.value.map((item) => ({
    ...item,
    status: 'failed',
    error: item.status === 'failed' && item.error ? item.error : reason,
    orphaned: true,
  }))
}

function countryLabel(iso: string | null | undefined): string {
  const code = normalizeIsoCode(iso)
  if (!code) return ''
  // Country-level contexts (candidate/confirm) always render the CN aggregate.
  if (isChinaRegionCode(code)) return '中国'
  return knownCountryNames.value[code] ?? code
}

function feedbackContent(payload: FeedbackPayload): string {
  switch (payload.type) {
    case 'confirm':
      return payload.iso ? `确认：${countryLabel(payload.iso)}` : '确认'
    case 'reject':
      return payload.iso ? `排除：${exclusionLabel(payload.iso)}` : '排除'
    case 'answer':
      return `补充线索：${dimensionLabel(payload.dimension ?? '')} = ${clueValueLabel(payload.value ?? '')}`
    case 'free_text':
      return (payload.text ?? '').trim()
    default:
      return ''
  }
}

function newAnalysis(): void {
  sessionEpoch.value += 1
  guided.clear()
  session.reset()
  cancelInFlightUi()
  releasePreviewUrl()
  selectedFile.value = null
  errorMessage.value = null
  uploadNotice.value = null
  resetTransientUi()
  orphanPending('原会话已变更，未收到服务器确认')
  lastOutcome.value = null
  historyOpen.value = false
  mobileEvidenceOpen.value = false
  rightCollapsed.value = false
  if (!isCompact.value) leftCollapsed.value = false
}

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return '仅支持 JPG、PNG、WebP 格式的图片'
  if (file.size <= 0) return '图片文件为空，请重新选择'
  if (file.size > MAX_UPLOAD_BYTES) return '图片不能超过 16 MB'
  return null
}

function onFile(file: File): void {
  const invalid = validateFile(file)
  if (invalid) {
    uploadNotice.value = invalid
    return
  }
  // A new file supersedes any in-flight request; the epoch guard drops its result.
  sessionEpoch.value += 1
  guided.clear()
  session.reset()
  cancelInFlightUi()
  resetTransientUi()
  releasePreviewUrl()
  orphanPending('原会话已变更，未收到服务器确认')
  selectedFile.value = file
  localPreviewUrl.value = URL.createObjectURL(file)
  uploadNotice.value = null
  errorMessage.value = null
  lastOutcome.value = null
}

async function upload(): Promise<void> {
  if (!selectedFile.value || uploading.value) return
  const epoch = ++sessionEpoch.value
  const file = selectedFile.value
  // A new upload supersedes city/feedback/point requests; their stale finally
  // blocks are epoch-guarded, so clear the shared busy flags here.
  cancelInFlightUi()
  orphanPending('上传新图片期间未收到服务器确认')
  uploading.value = true
  uploadNotice.value = null
  errorMessage.value = null
  try {
    const response = await api.upload(file)
    if (epoch !== sessionEpoch.value) return
    session.applyUpload(response)
    guided.clear()
    releasePreviewUrl()
    resetTransientUi()
    selectedIso.value = session.scores.value[0]?.iso ?? null
    lastOutcome.value = null
    rightCollapsed.value = false
    if (!isCompact.value) leftCollapsed.value = false
    void guided.start(response.session_id)
  } catch (caught) {
    if (epoch === sessionEpoch.value) setError(caught, '上传失败，请稍后重试')
  } finally {
    if (epoch === sessionEpoch.value) uploading.value = false
  }
}

interface CandidateSnapshot {
  clueSig: string
  rankSig: string
  exclusions: string[]
  cityCountryIso: string | null
  citySourceIso: string | null
}

function clueSignature(clues: readonly { dimension: string; value: string }[]): string {
  return clues.map((clue) => `${clue.dimension}\u0000${clue.value}`).join('|')
}

function rankingSignature(scores: readonly CountryScore[]): string {
  return scores.map((score) => score.iso).join(',')
}

function captureCandidateSnapshot(): CandidateSnapshot {
  return {
    clueSig: clueSignature(session.clues.value),
    rankSig: rankingSignature(session.scores.value),
    exclusions: [...session.exclusions.value],
    cityCountryIso: session.cityCountryIso.value,
    citySourceIso: citySourceIso.value,
  }
}

/**
 * Raw TW/HK/MO exclusions affect only that region; CN affects all members.
 * A merged (unfiltered) CN city list is only partially invalidated by a
 * regional exclusion, so it is refreshed in place instead of cleared.
 */
/** Raw region code of one city row: source fields first, then the row iso. */
function cityRowRegionCode(
  city: { iso?: string | null; source_iso?: string | null; source_region?: SourceRegionEntry | null },
): string {
  return (
    normalizeIsoCode(city.source_iso ?? '') ||
    (city.source_region ? entryCode(city.source_region) : '') ||
    normalizeIsoCode(city.iso)
  )
}

function cityScopeAffected(
  excluded: readonly string[],
  countryIso: string | null,
  sourceIso: string | null,
): 'none' | 'partial' | 'all' {
  const country = normalizeIsoCode(countryIso)
  if (!country) return 'none'
  const codes = excluded.map((code) => normalizeIsoCode(code))
  if (codes.includes(country)) return 'all'
  const source = normalizeIsoCode(sourceIso)
  if (source) return codes.includes(source) ? 'all' : 'none'
  return codes.some((code) => isChinaRegionCode(code)) ? 'partial' : 'none'
}

function handleCityStaleness(prev: CandidateSnapshot): void {
  const cityIso = prev.cityCountryIso
  if (!cityIso) return
  const country = normalizeIsoCode(cityIso)
  const affected = cityScopeAffected(session.exclusions.value, country, prev.citySourceIso)
  const canonical = normalizeCountryIso(country)
  const stillListed =
    session.scores.value.some((score) => score.iso === canonical) ||
    session.confirmedIso.value === canonical
  if (affected === 'all' || !stillListed) {
    session.clearCities()
    citySourceIso.value = null
    citySourceLabel.value = null
    cityNotice.value =
      affected === 'all'
        ? '所选国家/地区已被排除，城市候选已清空。'
        : '候选更新后该国家/地区已不在列表中，城市候选已清空。'
    return
  }
  const cluesChanged = clueSignature(session.clues.value) !== prev.clueSig
  const rankChanged = rankingSignature(session.scores.value) !== prev.rankSig
  if (affected === 'partial') {
    // A raw TW/HK/MO exclusion only removes that region: refilter the merged
    // list in place (mainland rows survive) and refresh explicitly.
    const before = session.cityScores.value.length
    const kept = session.cityScores.value.filter((city) => {
      const region = cityRowRegionCode(city)
      return !region || !session.exclusions.value.includes(region)
    })
    if (!kept.length) {
      session.clearCities()
      citySourceIso.value = null
      citySourceLabel.value = null
      cityNotice.value = '该区域范围已被排除，城市候选已清空。'
      return
    }
    session.setCityRows(kept)
    void requestCities(cityIso, prev.citySourceIso)
    cityNotice.value = `已按新的排除条件过滤城市候选（移除 ${before - kept.length} 个），正在刷新…`
    return
  }
  if (cluesChanged || rankChanged) {
    cityNotice.value = '线索或候选已更新，正在刷新城市候选…'
    void requestCities(cityIso, prev.citySourceIso)
  }
}

function addPending(payload: FeedbackPayload, content: string): PendingChatMessage {
  const pending: PendingChatMessage = {
    id: ++pendingNonce,
    role: 'user',
    content: content || '（反馈）',
    status: 'sending',
    error: null,
    payload,
    sessionId: session.sessionId.value,
    epoch: sessionEpoch.value,
  }
  pendingMessages.value = [...pendingMessages.value, pending]
  return pending
}

function updatePending(id: number, patch: Partial<PendingChatMessage>): void {
  pendingMessages.value = pendingMessages.value.map((item) =>
    item.id === id ? { ...item, ...patch } : item,
  )
}

function removePending(id: number): void {
  pendingMessages.value = pendingMessages.value.filter((item) => item.id !== id)
}

function finalizeFeedbackOutcome(response: FeedbackResponse, prev: CandidateSnapshot): void {
  const receipt = feedbackReceipt(response.messages ?? [], response.action ?? null)
  const parts: string[] = []
  if (receipt) {
    parts.push(receiptText(receipt))
  } else {
    // HTTP success alone is not understanding: without a receipt never claim it.
    parts.push('服务器已接收反馈（暂无解读回执）')
  }
  const notes: string[] = []
  if (response.done && response.iso) notes.push(`已确认国家：${countryLabel(response.iso)}`)
  const added = session.exclusions.value.filter((code) => !prev.exclusions.includes(code))
  if (added.length) notes.push(`已排除：${added.map((code) => exclusionLabel(code)).join('、')}`)
  if (clueSignature(session.clues.value) !== prev.clueSig) notes.push('线索已更新')
  if (rankingSignature(session.scores.value) !== prev.rankSig) notes.push('候选排序已更新')
  if (notes.length) parts.push(notes.join(' · '))
  lastOutcome.value = { tone: receipt?.tone ?? 'muted', text: parts.join(' · ') }
  // Only an applied (or receipt-less legacy) response may invalidate city rows:
  // unrecognized/unchanged feedback must keep valid results untouched.
  const status = receipt?.status ?? ''
  const applied = !receipt || status === 'applied' || status === 'understood' || status === 'accepted'
  if (applied) handleCityStaleness(prev)
}

async function submitFeedback(
  payload: FeedbackPayload,
  options: { content?: string; existing?: PendingChatMessage } = {},
): Promise<void> {
  const id = session.sessionId.value
  if (routeGuidedFeedback(payload)) return
  if (id === null || feedbackBusy.value) return
  const epoch = sessionEpoch.value
  const content = options.content ?? feedbackContent(payload)
  const pending = options.existing ?? addPending(payload, content)
  if (options.existing) {
    updatePending(pending.id, { status: 'sending', error: null, sessionId: id, epoch })
  }
  // Submit acknowledged: the draft is owned by the parent and restored on failure.
  if (payload.type === 'free_text') feedbackDraft.value = ''
  feedbackBusy.value = true
  lastOutcome.value = null
  errorMessage.value = null
  const snapshot = captureCandidateSnapshot()
  const userCountBefore = session.messages.value.filter((message) => message.role === 'user').length
  try {
    const response = await api.feedback(id, payload)
    if (epoch !== sessionEpoch.value) {
      updatePending(pending.id, { status: 'failed', error: '请求已被新的操作取代，未收到服务器确认' })
      return
    }
    const authoritative = Array.isArray(response.messages) ? response.messages : null
    session.applyFeedback(response)
    const userCountAfter = session.messages.value.filter((message) => message.role === 'user').length
    if (authoritative === null || userCountAfter <= userCountBefore) {
      // Server did not echo the user turn; keep the text visible exactly once.
      session.appendMessage({ role: 'user', content: pending.content, meta: { client_echo: true } })
    }
    removePending(pending.id)
    finalizeFeedbackOutcome(response, snapshot)
  } catch (caught) {
    if (epoch !== sessionEpoch.value) {
      updatePending(pending.id, { status: 'failed', error: '请求已被新的操作取代，未收到服务器确认' })
      return
    }
    const message = normalizeApiError(caught, undefined, '反馈发送失败').message
    updatePending(pending.id, { status: 'failed', error: message })
    if (payload.type === 'free_text' && payload.text && !feedbackDraft.value.trim()) {
      feedbackDraft.value = payload.text
    }
  } finally {
    if (epoch === sessionEpoch.value) feedbackBusy.value = false
  }
}

function retryPending(pending: PendingChatMessage): void {
  if (feedbackBusy.value || pending.orphaned) return
  if (guided.active.value && routeGuidedFeedback(pending.payload)) {
    removePending(pending.id)
    return
  }
  if (session.sessionId.value === null) {
    updatePending(pending.id, { status: 'failed', error: '当前没有会话，请先上传图片' })
    return
  }
  // Never rebind an old-photo bubble to a different session or epoch: the user
  // must resubmit it as a new message instead.
  if (pending.sessionId !== session.sessionId.value || pending.epoch !== sessionEpoch.value) {
    updatePending(pending.id, {
      status: 'failed',
      error: '原会话已变更，未发送到当前会话；请重新输入或删除',
      orphaned: true,
    })
    return
  }
  void submitFeedback(pending.payload, { content: pending.content, existing: pending })
}

function dismissPending(pending: PendingChatMessage): void {
  removePending(pending.id)
  if (pending.payload.type === 'free_text' && pending.payload.text && !feedbackDraft.value.trim()) {
    feedbackDraft.value = pending.payload.text
  }
}

/** Resolve an explicit UI action to the exact server candidate, never a rebuilt target. */
function currentGuidedCandidateTarget(target: LocationTarget): LocationTarget | null {
  if (!guided.active.value) return null
  const stage = guided.dialogue.value?.stage
  if (!stage || stage === 'resolved' || target.level !== stage) return null
  const candidate = guided.actionableCandidates.value.find(
    (item) => item.target.level === target.level && item.target.id === target.id,
  )
  return candidate?.target ?? null
}

function guidedCountryTargetForIso(iso: string | null | undefined): LocationTarget | null {
  if (guided.dialogue.value?.stage !== 'country') return null
  const code = normalizeIsoCode(iso ?? '')
  if (!code) return null
  const candidate = guided.actionableCandidates.value.find((item) => {
    if (item.target.level !== 'country') return false
    return [item.target.id, item.target.country_iso, item.target.source_iso].some(
      (value) => normalizeIsoCode(value ?? '') === code,
    )
  })
  return candidate?.target ?? null
}

function selectGuidedTarget(target: LocationTarget): void {
  if (busy.value) return
  const candidate = currentGuidedCandidateTarget(target)
  const key = candidate ? guidedTargetKey(candidate) : null
  if (key) selectedGuidedTargetKey.value = key
}

function confirmGuidedTarget(target: LocationTarget): void {
  if (busy.value || !guidedConfirmationReady.value) return
  const candidate = currentGuidedCandidateTarget(target)
  const key = candidate ? guidedTargetKey(candidate) : null
  if (!candidate || !key) return
  selectedGuidedTargetKey.value = key
  void guided.confirmLocation(candidate)
}

function rejectGuidedTarget(target: LocationTarget): void {
  if (busy.value || !guidedConfirmationReady.value) return
  const candidate = currentGuidedCandidateTarget(target)
  const key = candidate ? guidedTargetKey(candidate) : null
  if (!candidate || !key) return
  if (selectedGuidedTargetKey.value === key) selectedGuidedTargetKey.value = null
  void guided.rejectLocation(candidate)
}

watch(
  [guided.active, guided.dialogue, guided.candidates, guided.actionableCandidates],
  () => {
    const key = selectedGuidedTargetKey.value
    if (!key) return
    const stage = guided.dialogue.value?.stage
    if (!guided.active.value || !stage || stage === 'resolved') {
      selectedGuidedTargetKey.value = null
      return
    }
    const stillCurrent = guided.actionableCandidates.value.some(
      (candidate) => candidate.target.level === stage && guidedTargetKey(candidate.target) === key,
    )
    if (!stillCurrent) selectedGuidedTargetKey.value = null
  },
  { flush: 'sync' },
)

/** Chat/feedback buttons requesting confirmation must go through the dialog. */
function routeGuidedFeedback(payload: FeedbackPayload): boolean {
  if (!guided.active.value) return false
  if (busy.value) return true
  switch (payload.type) {
    case 'free_text':
      if (payload.text?.trim()) void guided.freeText(payload.text)
      break
    case 'confirm':
      if (payload.iso) {
        const target = guidedCountryTargetForIso(payload.iso)
        if (target) confirmGuidedTarget(target)
      }
      break
    case 'reject':
      if (payload.iso) {
        const target = guidedCountryTargetForIso(payload.iso)
        if (target) rejectGuidedTarget(target)
      }
      break
    case 'answer':
      void guided.answer(payload.value ?? '')
      break
  }
  return true
}

/** Chat/feedback buttons requesting confirmation must go through the dialog or guided event. */
function onFeedback(payload: FeedbackPayload): void {
  if (feedbackBusy.value || guided.applying.value || guided.loading.value) return
  if (routeGuidedFeedback(payload)) return
  // Confirm targets the canonical CN country; a reject keeps its raw region
  // scope (TW/HK/MO only excludes that region, CN excludes all members).
  let normalized: FeedbackPayload = payload
  if (payload.iso) {
    const iso = payload.type === 'reject' ? normalizeIsoCode(payload.iso) : normalizeCountryIso(payload.iso)
    normalized = { ...payload, iso }
  }
  if (normalized.type === 'confirm' && normalized.iso) {
    const candidate = session.scores.value.find((item) => item.iso === normalized.iso)
    if (candidate) {
      confirmCandidate.value = candidate
      return
    }
  }
  void submitFeedback(normalized, { content: feedbackContent(normalized) })
}

function doConfirm(): void {
  const candidate = confirmCandidate.value
  if (!candidate) return
  confirmCandidate.value = null
  if (guided.active.value) {
    const target = guidedCountryTargetForIso(candidate.iso)
    if (target) confirmGuidedTarget(target)
    return
  }
  const payload: FeedbackPayload = { type: 'confirm', iso: candidate.iso }
  void submitFeedback(payload, { content: feedbackContent(payload) })
}

function onCountryConfirm(score: CountryScore): void {
  if (guided.active.value) {
    if (busy.value || !guidedConfirmationReady.value) return
    const target = guidedCountryTargetForIso(score.iso)
    if (target) confirmGuidedTarget(target)
    return
  }
  confirmCandidate.value = score
}

function onCountryReject(score: CountryScore): void {
  if (guided.active.value) {
    if (busy.value || !guidedConfirmationReady.value) return
    const target = guidedCountryTargetForIso(score.iso)
    if (target) rejectGuidedTarget(target)
    return
  }
  onFeedback({ type: 'reject', iso: score.iso })
}

function resetRegionAnalysisState(): void {
  regionRequestNonce += 1
  regionRequests.value = []
  immediateRegionReports.value = []
  selectedRegionReportId.value = null
  guidedDraftText.value = null
}

function markRegionRequest(
  requestId: string,
  patch: Partial<Omit<RegionAnalysisRecord, 'id' | 'request' | 'sessionId' | 'epoch'>>,
): void {
  regionRequests.value = regionRequests.value.map((record): RegionAnalysisRecord =>
    record.id === requestId ? { ...record, ...patch } : record,
  )
}

function removeRegionRequest(requestId: string): void {
  regionRequests.value = regionRequests.value.filter((record) => record.id !== requestId)
}

function onRegionSelectionChange(): void {
  // A new ROI supersedes any prior in-flight payload; late responses are ignored.
  regionRequestNonce += 1
  regionRequests.value = regionRequests.value.map((record): RegionAnalysisRecord =>
    record.status === 'pending'
      ? { ...record, status: 'failed', error: '请求已被新的区域选择取代，未采用。' }
      : record,
  )
  pointStatus.value = '已选择新的区域，点击“分析选定区域”后再提交。'
  pointStatusKind.value = 'idle'
}

async function analyzeRegionRequest(request: AnalyzePointRequest): Promise<void> {
  const id = session.sessionId.value
  const normalized = normalizeAnalyzePointRequest(request)
  if (id === null || normalized === null || pointBusy.value || guided.loading.value || guided.applying.value) {
    return
  }
  const requestId = normalized.client_request_id || newRegionRequestId()
  normalized.client_request_id = requestId
  const existing = regionRequests.value.find((record) => record.id === requestId)
  if (existing && (existing.sessionId !== id || existing.epoch !== sessionEpoch.value)) {
    markRegionRequest(requestId, { status: 'failed', error: '原会话已变更，请在当前图片上重新选择区域。' })
    return
  }

  const epoch = sessionEpoch.value
  const requestNonce = ++regionRequestNonce
  const snapshot = captureCandidateSnapshot()
  const pendingRecord: RegionAnalysisRecord = {
    id: requestId,
    request: normalized,
    status: 'pending',
    error: null,
    sessionId: id,
    epoch,
  }
  regionRequests.value = [
    pendingRecord,
    ...regionRequests.value.filter((record) => record.id !== requestId),
  ]
  pointBusy.value = true
  pointStatus.value = '正在分析所选区域…'
  pointStatusKind.value = 'loading'
  errorMessage.value = null

  const stale = (): boolean => epoch !== sessionEpoch.value || requestNonce !== regionRequestNonce
  try {
    const response = await api.analyzePoint(id, normalized)
    if (stale()) {
      markRegionRequest(requestId, { status: 'failed', error: '请求已被新的区域选择取代，未采用。' })
      return
    }

    session.applyPointAnalysis(response)
    clickPoints.value = [...clickPoints.value, { x: normalized.x, y: normalized.y }]
    handleCityStaleness(snapshot)
    const newClueCount = response.new_clues?.length ?? 0
    const report =
      normalizeRegionReport(response.focus_region)
      ?? regionReportsFromMessages(response.messages ?? []).at(-1)
      ?? null
    if (report) {
      immediateRegionReports.value = mergeRegionReports(immediateRegionReports.value, [report])
      selectedRegionReportId.value = report.id
    } else {
      selectedRegionReportId.value = null
    }

    const applied = response.applied !== false
    let completionText: string
    if (!report) {
      completionText = newClueCount
        ? `分析完成，发现 ${newClueCount} 条新线索；该响应没有区域报告，沿用原有线索。`
        : '分析完成，没有新增线索；该响应没有区域报告，沿用原有线索。'
    } else if (report.dedupe === 'duplicate_request' || report.dedupe === 'duplicate_region') {
      completionText = `分析完成：${report.summary || '该区域已有报告'}（未重复写入证据）`
    } else if (!applied) {
      completionText = `分析完成：${report.summary || '未发现新增证据'}（报告已保留）`
    } else {
      completionText = `分析完成：${report.summary || `发现 ${newClueCount} 条新线索`}`
    }

    // Evidence mutates in the same session; keep global busy true until the
    // authoritative guided snapshot has been reloaded.
    if (guided.active.value) {
      const reloaded = await guided.reload()
      if (stale()) {
        markRegionRequest(requestId, { status: 'failed', error: '请求已被新的区域选择取代，未采用。' })
        return
      }
      if (!reloaded) {
        const syncError = `区域分析已返回，但引导状态同步失败：${guided.error.value ?? '请重试'}`
        markRegionRequest(requestId, { status: 'failed', error: syncError })
        pointStatus.value = syncError
        pointStatusKind.value = 'error'
        return
      }
    }

    removeRegionRequest(requestId)
    pointStatus.value = completionText
    pointStatusKind.value = 'success'
  } catch (caught) {
    if (stale()) {
      markRegionRequest(requestId, { status: 'failed', error: '请求已被新的区域选择取代，未采用。' })
      return
    }
    const message = normalizeApiError(caught, undefined, '局部区域分析失败').message
    markRegionRequest(requestId, { status: 'failed', error: message })
    pointStatus.value = `分析失败：${message}`
    pointStatusKind.value = 'error'
    setError(caught, '局部区域分析失败')
  } finally {
    if (epoch === sessionEpoch.value) pointBusy.value = false
  }
}

function retryRegionRequest(request: AnalyzePointRequest): void {
  if (pointBusy.value || guided.loading.value || guided.applying.value) return
  void analyzeRegionRequest(request)
}

function useRegionQuestion(question: RegionQuestion): void {
  const text = question.text.trim()
  if (!text) return
  guidedDraftText.value = text
  if (!guided.active.value) feedbackDraft.value = text
}

async function onGuidedFreeText(text: string): Promise<void> {
  const submitted = text.trim()
  if (!submitted) return
  const nonce = ++guidedFreeTextNonce
  const sessionId = session.sessionId.value
  guidedFreeTextSubmission.value = { text: submitted, status: 'pending' }
  const applied = await guided.freeText(submitted)
  if (nonce !== guidedFreeTextNonce || session.sessionId.value !== sessionId) return
  if (!applied) {
    guidedFreeTextSubmission.value = {
      text: submitted,
      status: 'failed',
      error: guided.error.value ?? guided.conflict.value ?? '提交失败，请重试。',
    }
    return
  }

  const outcome = guidedFreeTextReceiptOutcome(guided.receipt.value)
  if (outcome.status === 'failed') {
    guidedFreeTextSubmission.value = {
      text: submitted,
      status: 'failed',
      error: outcome.error ?? '没有识别到新的线索，请修改后重试。',
    }
    return
  }

  guidedDraftText.value = null
  // Preserve the server's distinction between an applied mutation and an
  // acknowledged pre-existing clue; never relabel unchanged as applied.
  guidedFreeTextSubmission.value = { text: submitted, status: outcome.status }
}

async function requestCities(iso: string, sourceIso?: string | null): Promise<void> {
  const target = normalizeIsoCode(iso)
  const id = session.sessionId.value
  if (!target || id === null || cityBusy.value) return
  const source = normalizeIsoCode(sourceIso ?? '') || null
  const epoch = sessionEpoch.value
  const requestNonce = ++cityRequestNonce
  cityBusy.value = true
  cityNotice.value = null
  try {
    const response = await api.cities(id, target, source)
    if (epoch !== sessionEpoch.value || requestNonce !== cityRequestNonce) return
    session.applyCities(response)
    citySourceIso.value = source
    const responseSource = normalizeIsoCode(response.source_iso ?? '') || source
    citySourceLabel.value =
      (typeof response.source_region === 'string' ? response.source_region.trim() : null) || (responseSource ? CHINA_REGION_LABELS[responseSource] ?? null : null)
    cityNotice.value = [response.note, response.clue_note].filter(Boolean).join('；') || null
    if (response.status === 'unavailable') {
      cityNotice.value ||= '模型服务当前不可用，暂时无法检索城市候选；可稍后重试。'
    } else if (response.status === 'no_cities' || !response.city_scores.length) {
      cityNotice.value ||= source
        ? '该区域暂无城市候选；可切换区域范围或稍后重试。'
        : '该国家/地区暂无城市候选；可切换重试。'
    }
  } catch {
    if (epoch !== sessionEpoch.value || requestNonce !== cityRequestNonce) return
    // Degrade inline: a failed city lookup must never raise a global error banner.
    session.clearCities()
    citySourceLabel.value = null
    cityNotice.value = '城市候选服务不可用，可稍后重试。'
  } finally {
    if (epoch === sessionEpoch.value && requestNonce === cityRequestNonce) cityBusy.value = false
  }
}
</script>

<template>
  <div class="app-shell immersive-app" :class="{ 'is-mobile-layout': isMobile, 'is-compact-layout': isCompact }">
    <header ref="commandBarRef" class="command-bar" aria-label="地理定位命令栏" :inert="backgroundInert">
      <div class="command-brand">
        <span class="command-brand__mark" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">
            <circle cx="12" cy="12" r="8.4" />
            <path d="M3.8 12h16.4M12 3.6c2.4 2.5 3.6 5.3 3.6 8.4S14.4 17.9 12 20.4c-2.4-2.5-3.6-5.3-3.6-8.4S9.6 6.1 12 3.6z" />
          </svg>
        </span>
        <span class="command-brand__copy">
          <h1 class="command-brand__title">地理定位推理台</h1>
          <span class="command-brand__sub">证据 · 推演 · 人工确认</span>
        </span>
      </div>

      <nav class="session-breadcrumb" aria-label="当前推理路径">
        <span class="breadcrumb-step">
          <span class="breadcrumb-index">01</span>
          <span class="breadcrumb-label">国家</span>
          <strong :title="countryBreadcrumb">{{ countryBreadcrumb }}</strong>
        </span>
        <span class="breadcrumb-arrow" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M5 12h14M14 7l5 5-5 5" />
          </svg>
        </span>
        <span class="breadcrumb-step breadcrumb-step--city">
          <span class="breadcrumb-index">02</span>
          <span class="breadcrumb-label">城市</span>
          <strong :title="cityBreadcrumb">{{ cityBreadcrumb }}</strong>
        </span>
      </nav>

      <div class="command-session" :class="liveStatus.tone" :title="sessionStatus">
        <span class="session-dot" :class="{ active: hasSession, busy }" />
        <span class="command-session__text">{{ sessionStatus }}</span>
      </div>

      <div class="focus-switch" role="group" aria-label="界面聚焦模式">
        <button
          v-for="mode in FOCUS_MODES"
          :key="mode.key"
          type="button"
          class="focus-mode-btn"
          :class="{ active: focusMode === mode.key }"
          :aria-pressed="focusMode === mode.key"
          :aria-label="mode.label"
          :title="mode.hint"
          @click="setFocusMode(mode.key)"
        >
          <svg v-if="mode.key === 'collab'" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
            <path d="M8 9h6M8 13h4M6 18l-3 2 1-4a7 7 0 1 1 12 0" />
            <path d="M13 10h5M13 14h3M18 18l3 2-1-4a6 6 0 0 0-4-5" />
          </svg>
          <svg v-else-if="mode.key === 'map'" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
            <path d="m3 6 5-2 8 3 5-2v14l-5 2-8-3-5 2zM8 4v14M16 7v14" />
          </svg>
          <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="2.5" />
            <circle cx="9" cy="9" r="1.7" />
            <path d="m21 15-5-5L6 20" />
          </svg>
          <span class="focus-label">{{ mode.label }}</span>
        </button>
      </div>

      <button
        type="button"
        class="command-action"
        :class="{ active: historyOpen }"
        :aria-expanded="historyOpen"
        aria-controls="reasoning-history-drawer"
        :disabled="!historyAllowed"
        :title="historyAllowed ? '打开推理记录' : '观察核对完成并解锁候选后可查看推理记录'"
        @click="openHistory"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
          <path d="M3 3v5h5M12 7v5l3 2" />
        </svg>
        <span class="command-action__label">历史</span>
      </button>

      <button type="button" class="command-action command-action--strong" @click="newAnalysis">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <path d="M12 5v14M5 12h14" />
        </svg>
        <span class="command-action__label">新建分析</span>
      </button>
    </header>

    <main class="map-stage" aria-label="沉浸式候选地图" :inert="backgroundInert">
      <div ref="mapHostRef" class="map-host">
        <GeoMap
          :scores="guidedCountryScores"
          :confirmed-iso="session.confirmedIso.value"
          :city-scores="visibleCityScores"
          :city-country-iso="session.cityCountryIso.value"
          :selected-iso="selectedIso"
          :guided="guided.active.value"
          :guided-stage="guidedMapStage"
          :guided-candidates="guidedVisibleCandidates"
          :guided-candidates-visible="guidedCandidateSurfaceVisible"
          :guided-confirmed="guided.dialogue.value?.confirmed ?? null"
          :selected-guided-target-key="selectedGuidedTargetKey"
          :guided-confirmation-ready="guidedConfirmationReady"
          :busy="busy"
          immersive
          :viewport-insets="mapInsets"
          @confirm="onCountryConfirm"
          @select-country="selectedIso = $event"
          @select-guided-target="selectGuidedTarget"
          @confirm-guided-target="confirmGuidedTarget"
        />
      </div>
    </main>

    <transition name="fade">
      <div
        v-if="isMobile && mobileEvidenceOpen"
        class="mobile-evidence-backdrop"
        aria-hidden="true"
        @click="closeMobileEvidence"
      />
    </transition>

    <aside
      v-show="evidenceAnchorVisible"
      ref="evidenceDockRef"
      class="dock-anchor evidence-dock-anchor"
      :class="{ 'is-mobile-open': isMobile && mobileEvidenceOpen, 'is-collapsed': showEvidenceThumbnail }"
      :inert="historyOpen"
      aria-label="照片与证据"
    >
      <button
        v-if="showEvidenceThumbnail"
        type="button"
        class="dock-thumbnail evidence-thumbnail"
        aria-label="展开证据台"
        @click="expandEvidenceDock"
      >
        <span class="dock-thumbnail__media">
          <img v-if="previewUrl" :src="previewUrl" alt="" />
          <span v-else class="dock-thumbnail__empty" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <rect x="3" y="4" width="18" height="16" rx="2.5" />
              <path d="m21 15-5-5L6 20" />
            </svg>
          </span>
          <span class="dock-thumbnail__badge">{{ session.imageUrl.value ? 'PHOTO' : 'UPLOAD' }}</span>
        </span>
        <span class="dock-thumbnail__copy">
          <strong>证据台</strong>
          <small>{{ clueCount }} 条线索</small>
        </span>
      </button>

      <section
        v-else
        ref="mobileEvidenceDialogRef"
        class="floating-dock evidence-dock"
        :role="isMobile ? 'dialog' : undefined"
        :aria-modal="isMobile ? 'true' : undefined"
        :tabindex="isMobile ? -1 : undefined"
        aria-labelledby="evidence-dock-title"
      >
        <header class="dock-head">
          <div class="dock-heading">
            <span class="section-kicker">EVIDENCE</span>
            <h2 id="evidence-dock-title">照片与证据</h2>
          </div>
          <div class="dock-head__actions">
            <span v-if="session.imageUrl.value" class="dock-state-chip">{{ regionReports.length }} 份区域报告</span>
            <button
              v-if="!isMobile"
              type="button"
              class="icon-btn"
              aria-label="收起证据台"
              title="收起为缩略图"
              @click="collapseEvidenceDock"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <path d="m15 18-6-6 6-6" />
              </svg>
            </button>
            <button
              v-else
              type="button"
              class="icon-btn"
              data-modal-autofocus
              aria-label="关闭照片与线索浮层"
              @click="closeMobileEvidence"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <path d="M6 6l12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        </header>

        <div v-if="isMobile" class="evidence-mobile-tabs" role="tablist" aria-label="证据视图">
          <button type="button" role="tab" :aria-selected="mobileEvidenceTab === 'photo'" :class="{ active: mobileEvidenceTab === 'photo' }" @click="openMobileEvidence('photo')">照片</button>
          <button type="button" role="tab" :aria-selected="mobileEvidenceTab === 'clues'" :class="{ active: mobileEvidenceTab === 'clues' }" @click="openMobileEvidence('clues')">线索 <span>{{ clueCount }}</span></button>
          <button type="button" role="tab" :aria-selected="mobileEvidenceTab === 'references'" :class="{ active: mobileEvidenceTab === 'references' }" @click="openMobileEvidence('references')">参考</button>
        </div>

        <div class="evidence-scroll">
          <section v-if="!isMobile || mobileEvidenceTab === 'photo'" class="evidence-section evidence-section--photo">
            <div class="dock-section-heading">
              <div>
                <span class="section-kicker">PHOTO</span>
                <h3>原始照片与区域分析</h3>
              </div>
              <span v-if="session.imageUrl.value" class="dock-muted">点击照片可更换</span>
            </div>

            <UploadPanel
              :preview-url="previewUrl"
              :click-points="clickPoints"
              :disabled="uploading"
              @file="onFile"
            />
            <div v-if="uploadNotice" class="inline-error">{{ uploadNotice }}</div>
            <div v-if="selectedFile" class="file-chip">
              <span>{{ selectedFile.name }}</span>
              <span class="state">{{ hasSession && !localPreviewUrl ? '已分析' : '待上传' }}</span>
            </div>
            <button
              v-if="selectedFile && (!hasSession || localPreviewUrl)"
              type="button"
              class="primary-btn upload-btn"
              :disabled="uploading"
              @click="upload"
            >
              {{ uploading ? '上传并分析中…' : '上传并分析' }}
            </button>

            <div class="evidence-actions">
              <button
                type="button"
                class="ghost-btn evidence-viewer-btn"
                :disabled="!session.imageUrl.value"
                @click="openImageViewer"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <circle cx="11" cy="11" r="6.5" />
                  <path d="m16 16 4 4M8 11h6M11 8v6" />
                </svg>
                打开区域分析
              </button>
              <span class="evidence-footnote">点选 / 框选后人工确认；坐标不代表照片 GPS</span>
            </div>
          </section>

          <section v-if="!isMobile || mobileEvidenceTab === 'clues'" class="evidence-section evidence-section--clues">
            <div class="dock-section-heading">
              <div>
                <span class="section-kicker">OBSERVATIONS</span>
                <h3>线索卡 <span class="dock-count">{{ clueCount }}</span></h3>
              </div>
              <span class="dock-muted">按证据强度排序</span>
            </div>
            <ClueList
              :clues="session.clues.value"
              :loading="uploading"
              :recent-clue-keys="recentClueKeys"
              @locate-clue="openFocusedClue"
            />
          </section>

          <section v-if="!isMobile || mobileEvidenceTab === 'references'" class="evidence-section evidence-section--references">
            <div class="dock-section-heading">
              <div>
                <span class="section-kicker">REFERENCES</span>
                <h3>参考街景</h3>
              </div>
              <span class="dock-muted">相似度仅作诊断</span>
            </div>
            <CandidatePanel
              surface="references"
              embedded
              :scores="guidedCountryScores"
              :confirmed-iso="session.confirmedIso.value"
              :selected-iso="selectedIso"
              :exclusions="exclusionChips"
              :busy="busy"
              :guided-mode="guided.active.value"
              :guided-country-allowed="!guided.active.value || guided.dialogue.value?.stage === 'country'"
              :guided-candidates="guidedVisibleCandidates"
              :guided-candidates-unlocked="guidedCandidateSurfaceVisible"
              :guided-confirmation-ready="guidedConfirmationReady"
              :guided-data-ready="guidedDataReady"
              :guided-stage="guided.dialogue.value?.stage ?? null"
              :city-scores="visibleCityScores"
              :city-country-iso="session.cityCountryIso.value"
              :cities-status="session.citiesStatus.value"
              :country-options="countryOptions"
              :city-busy="cityBusy"
              :city-notice="cityNotice"
              :city-source-iso="citySourceIso"
              :city-source-label="citySourceLabel"
            />
          </section>
        </div>
      </section>
    </aside>

    <aside
      v-show="reasoningAnchorVisible"
      ref="reasoningDockRef"
      class="dock-anchor reasoning-dock-anchor"
      :class="{ 'is-wide': reasoningWide, 'mobile-task-sheet': isMobile, 'is-collapsed': showReasoningTaskChip, 'is-guided-loading': guided.active.value && !guided.dialogue.value }"
      :data-sheet-state="mobileSheetState"
      :inert="backgroundInert"
      aria-label="推理任务"
    >
      <button
        v-if="showReasoningTaskChip"
        type="button"
        class="reasoning-task-chip"
        aria-label="展开推理任务"
        @click="expandReasoningDock"
      >
        <span class="reasoning-task-chip__icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">
            <path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z" />
            <path d="M8 10h8M8 14h5" />
          </svg>
        </span>
        <span class="reasoning-task-chip__copy">
          <span>推理任务</span>
          <strong>{{ guided.active.value ? '引导定位' : '协同推理' }}</strong>
          <small>{{ liveStatus.text }}</small>
        </span>
        <span class="live-dot" :class="liveStatus.tone" aria-hidden="true" />
      </button>

      <section v-show="!showReasoningTaskChip" class="floating-dock reasoning-dock" aria-labelledby="reasoning-dock-title">
        <header class="dock-head reasoning-dock__head">
          <div class="dock-heading">
            <span class="section-kicker">REASONING</span>
            <h2 id="reasoning-dock-title">{{ guided.active.value ? '引导推演' : '协同推理' }}</h2>
            <span class="dock-muted">{{ session.messages.value.length }} 条记录</span>
          </div>
          <div class="dock-head__actions">
            <div v-if="isMobile" class="sheet-state-switch" role="group" aria-label="任务面板高度">
              <button type="button" :class="{ active: mobileSheetState === 'peek' }" :aria-pressed="mobileSheetState === 'peek'" title="精简高度，保留输入框" @click="setMobileSheetState('peek')">精简</button>
              <button type="button" :class="{ active: mobileSheetState === 'half' }" :aria-pressed="mobileSheetState === 'half'" title="半屏展开任务" @click="setMobileSheetState('half')">半屏</button>
              <button type="button" :class="{ active: mobileSheetState === 'full' }" :aria-pressed="mobileSheetState === 'full'" title="全屏展开任务" @click="setMobileSheetState('full')">全屏</button>
            </div>
            <template v-else>
              <button
                type="button"
                class="icon-btn width-toggle"
                :class="{ active: reasoningWide }"
                :aria-pressed="reasoningWide"
                :aria-label="reasoningWide ? '还原推理面板宽度' : '加宽推理面板'"
                :title="reasoningWide ? '还原宽度' : '加宽到约 560px'"
                @click="toggleReasoningWidth"
              >
                <svg v-if="!reasoningWide" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <path d="M4 5v14M20 5v14M8 12h8M11 9l3 3-3 3" />
                </svg>
                <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <path d="M4 5v14M20 5v14M8 12h8M13 9l-3 3 3 3" />
                </svg>
              </button>
              <button
                type="button"
                class="icon-btn"
                aria-label="收起推理面板"
                title="收起为任务卡"
                @click="collapseReasoningDock"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <path d="m9 6 6 6-6 6" />
                </svg>
              </button>
            </template>
          </div>
        </header>

        <div class="reasoning-live" :class="liveStatus.tone" role="status" aria-live="polite">
          <span class="live-dot" aria-hidden="true" />
          <span>{{ liveStatus.text }}</span>
          <span v-if="questionPending" class="chat-alert">待回答</span>
        </div>

        <div class="reasoning-body" :class="{ 'has-legacy-candidates': !guided.active.value }">
          <div v-if="inferenceNotice" class="inference-notice reasoning-notice" role="status">
            {{ inferenceNotice }}
          </div>

          <div class="reasoning-grid">
            <div class="reasoning-primary">
              <GuidedDialoguePanel
                v-if="guided.active.value"
                :dialogue="guided.dialogue.value"
                :candidates="guided.topCandidates.value"
                :action="guided.action.value"
                :confidence="guided.confidence.value"
                :confidence-shortcut="guided.confidenceShortcut.value"
                :confidence-candidates="guided.confidenceCandidates.value"
                :loading="guided.loading.value"
                :applying="guided.applying.value"
                :external-busy="uploading || feedbackBusy || pointBusy || cityBusy"
                :error="guided.error.value"
                :conflict="guided.conflict.value"
                :can-retry="guided.canRetry.value"
                :receipt="guided.receipt.value"
                :selected-guided-target-key="selectedGuidedTargetKey"
                :suggested-free-text="guidedDraftText"
                :free-text-submission="guidedFreeTextSubmission"
                :phase="guided.phase.value"
                :candidates-unlocked="guidedRankingsUnlocked"
                :can-review-back="guided.canReviewBack.value"
                :location-confirmation-ready="guidedConfirmationReady"
                :recent-clue-ids="guided.recentClueIds.value"
                @select-location="selectGuidedTarget"
                @confirm-location="confirmGuidedTarget"
                @reject-location="rejectGuidedTarget"
                @confirm-clue="guided.confirmClue"
                @reject-clue="guided.rejectClue"
                @answer="(payload) => guided.answer(payload.value)"
                @unknown="guided.unknown()"
                @retract-target="guided.retractTarget"
                @retract-clue="guided.retractClue"
                @backtrack="guided.backtrack"
                @next-phase="guided.nextPhase()"
                @back-to-review="guided.returnToReview()"
                @review-back="guided.reviewBack()"
                @free-text="onGuidedFreeText"
                @retry="guided.retry"
              />

              <div v-else class="legacy-reasoning">
                <div class="legacy-thread">
                  <ChatLog
                    :messages="session.messages.value"
                    :pending="pendingMessages"
                    :loading="pointBusy || feedbackBusy"
                    @retry="retryPending"
                    @dismiss="dismissPending"
                  />
                </div>
                <FeedbackBar
                  v-model:draft="feedbackDraft"
                  :action="session.action.value"
                  :busy="feedbackBusy || uploading"
                  :disabled="!hasSession"
                  @feedback="onFeedback"
                />
              </div>
            </div>

            <CandidatePanel
              v-if="!guided.active.value"
              surface="candidates"
              class="legacy-candidate-panel"
              :scores="guidedCountryScores"
              :confirmed-iso="session.confirmedIso.value"
              :selected-iso="selectedIso"
              :exclusions="exclusionChips"
              :busy="busy"
              :guided-mode="false"
              :guided-country-allowed="true"
              :guided-candidates="guidedVisibleCandidates"
              :guided-candidates-unlocked="guidedCandidateSurfaceVisible"
              :guided-confirmation-ready="guidedConfirmationReady"
              :guided-data-ready="guidedDataReady"
              :guided-stage="guided.dialogue.value?.stage ?? null"
              :city-scores="visibleCityScores"
              :city-country-iso="session.cityCountryIso.value"
              :cities-status="session.citiesStatus.value"
              :country-options="countryOptions"
              :city-busy="cityBusy"
              :city-notice="cityNotice"
              :city-source-iso="citySourceIso"
              :city-source-label="citySourceLabel"
              @confirm="onCountryConfirm"
              @reject="onCountryReject"
              @select-country="selectedIso = $event"
              @select-guided-target="selectGuidedTarget"
              @confirm-guided-target="confirmGuidedTarget"
              @reject-guided-target="rejectGuidedTarget"
              @request-cities="requestCities"
            />
          </div>
        </div>
      </section>
    </aside>

    <transition name="drawer">
      <div v-if="historyOpen" class="history-layer">
        <div class="drawer-backdrop" @click="closeHistory" />
        <aside
          id="reasoning-history-drawer"
          ref="historyDialogRef"
          class="history-drawer"
          role="dialog"
          aria-modal="true"
          tabindex="-1"
          aria-labelledby="history-drawer-title"
        >
          <header class="history-drawer__head">
            <div>
              <span class="section-kicker">HISTORY</span>
              <h2 id="history-drawer-title">推理记录</h2>
              <p>完整对话、回执与待发送反馈</p>
            </div>
            <div class="history-drawer__actions">
              <span>{{ session.messages.value.length }} 条</span>
              <button type="button" class="icon-btn" data-modal-autofocus aria-label="关闭推理记录" @click="closeHistory">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <path d="M6 6l12 12M18 6 6 18" />
                </svg>
              </button>
            </div>
          </header>
          <div class="history-drawer__body">
            <ChatLog
              :messages="session.messages.value"
              :pending="pendingMessages"
              :loading="pointBusy || feedbackBusy"
              tabindex="0"
              role="region"
              aria-label="完整推理记录，可滚动查看"
              @retry="retryPending"
              @dismiss="dismissPending"
            />
          </div>
        </aside>
      </div>
    </transition>

    <transition name="fade">
      <div v-if="errorMessage" class="error-banner" role="alert" :inert="backgroundInert">
        <span>{{ errorMessage }}</span>
        <button type="button" aria-label="关闭错误提示" @click="errorMessage = null">×</button>
      </div>
    </transition>

    <ConfirmLocationDialog
      v-if="confirmCandidate"
      :country="confirmCandidate"
      :subregion="confirmSubregion"
      :busy="feedbackBusy || guided.applying.value"
      @confirm="doConfirm"
      @cancel="confirmCandidate = null"
    />

    <ImageViewer
      v-if="viewerOpen && session.imageUrl.value"
      :image-url="session.imageUrl.value"
      :click-points="clickPoints"
      :analyzing="pointBusy"
      :status="pointStatus"
      :status-kind="pointStatusKind"
      :region-reports="regionReports"
      :selected-region-report-id="selectedRegionReportId"
      :region-requests="regionRequests"
      :disabled="!hasSession || busy"
      :focused-clue="focusedClue"
      @analyze="analyzeRegionRequest"
      @select-region="selectedRegionReportId = $event"
      @retry-region="retryRegionRequest"
      @ask-question="useRegionQuestion"
      @selection-change="onRegionSelectionChange"
      @close="viewerOpen = false"
    />
  </div>
</template>

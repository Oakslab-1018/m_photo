<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
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
import { useGuidedSubmissions } from '@/composables/useGuidedSubmissions'
import { useWorkspaceLayout } from '@/composables/useWorkspaceLayout'
import { clueIdentity, clueValueLabel, countGroupedClues, dimensionLabel } from '@/utils/clues'
import { isQuestionAction } from '@/utils/dialogue'
import {
  guidedQuestionMessage,
  guidedStageTitle,
  guidedTargetKey,
  pendingQuestionClue,
  safeGuidedText,
  targetLabel,
} from '@/utils/guided'
import { feedbackReceipt, receiptText } from '@/utils/feedback'
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
import CommandBar from '@/components/CommandBar.vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import EvidenceDock from '@/components/EvidenceDock.vue'
import GeoMap from '@/components/GeoMap.vue'
import HistoryDrawer from '@/components/HistoryDrawer.vue'
import ReasoningDock from '@/components/ReasoningDock.vue'
import ConfirmLocationDialog from '@/components/ConfirmLocationDialog.vue'
import ImageViewer from '@/components/ImageViewer.vue'
import type {
  CandidatePanelBindings,
  GuidedDialogueBindings,
  LiveStatus,
} from '@/components/uiTypes'

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
const pointStatus = ref('在照片上点选或框选一处，然后点“分析选定区域”。')
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
const lastOutcome = ref<LiveStatus | null>(null)
const citySourceIso = ref<string | null>(null)
const citySourceLabel = ref<string | null>(null)
const knownCountryNames = ref<Record<string, string>>({})
const regionRequests = ref<RegionAnalysisRecord[]>([])
const immediateRegionReports = ref<RegionReport[]>([])
const selectedRegionReportId = ref<string | null>(null)
const guidedDraftText = ref<string | null>(null)

/** Monotonic session generation: stale responses can never overwrite newer state. */
const sessionEpoch = ref(0)
/** Monotonic city-request nonce: older city responses must not replace newer country picks. */
let cityRequestNonce = 0
let pendingNonce = 0
let regionRequestNonce = 0

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

const {
  guidedFreeTextSubmission,
  observationAdoption,
  observationAdoptionRetryable,
  resetGuidedFreeTextSubmission,
  resetObservationAdoption,
  onGuidedFreeText,
  onAdoptObservation,
  retryObservationAdoption,
  retryGuidedOperation,
} = useGuidedSubmissions({
  sessionId: session.sessionId,
  sessionEpoch,
  busy: () => busy.value,
  regionReports: () => regionReports.value,
  guided,
  guidedDraftText,
})
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

// This composable installs an immediate history-gate watcher, so initialize it
// only after the guided candidate surface is available.
const {
  commandBarRef,
  mapHostRef,
  evidenceDockRef,
  reasoningDockRef,
  historyDialogRef,
  mobileEvidenceDialogRef,
  focusMode,
  reasoningWide,
  mobileSheetState,
  mobileEvidenceOpen,
  mobileEvidenceTab,
  historyOpen,
  mapInsets,
  isMobile,
  isCompact,
  historyAllowed,
  backgroundInert,
  evidenceAnchorVisible,
  reasoningAnchorVisible,
  showEvidenceThumbnail,
  showReasoningTaskChip,
  setFocusMode,
  expandEvidenceDock,
  collapseEvidenceDock,
  expandReasoningDock,
  collapseReasoningDock,
  toggleReasoningWidth,
  setMobileSheetState,
  openMobileEvidence,
  closeMobileEvidence,
  closeMobileEvidenceForViewer,
  openHistory,
  closeHistory,
  restoreDefaultDocks,
} = useWorkspaceLayout({
  historyAllowed: () => !guided.active.value || guidedCandidateSurfaceVisible.value,
})

const layoutElementRefs = {
  mapHost: mapHostRef,
  commandBar: commandBarRef,
  evidenceDock: evidenceDockRef,
  reasoningDock: reasoningDockRef,
  historyDialog: historyDialogRef,
  mobileEvidenceDialog: mobileEvidenceDialogRef,
}
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
  if (uploading.value) return '正在分析照片…'
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return guided.loading.value ? '正在准备…' : '加载失败，请重试'
    if (current.stage === 'resolved') {
      return `已完成 · ${targetLabel(current.confirmed.city)}`
    }
    if (!guidedCandidateSurfaceVisible.value) {
      if (guided.reviewReady.value) return '核对完成'
      const review = guided.reviewProgress.value
      return review
        ? `核对观察 ${review.current}/${review.total}`
        : `还要核对 ${guided.pendingClueCount.value} 条`
    }
    if (!guidedConfirmationReady.value) {
      return `${guidedStageTitle(current.stage)} · 有新观察待核对`
    }
    return `${guidedStageTitle(current.stage)} · ${guidedVisibleCandidates.value.length} 个候选`
  }
  if (session.confirmedIso.value) {
    const country = session.scores.value.find((item) => item.iso === session.confirmedIso.value)
    const scope = country ? regionScopeLabel(country) : null
    const name = country ? displayCountryName(country) : session.confirmedIso.value
    return `已确认 · ${name}${scope ? `（${scope}）` : ''}`
  }
  if (!hasSession.value) return '未开始'
  return `${session.scores.value.length} 个国家候选`
})

const countryBreadcrumb = computed(() => {
  const target = guided.dialogue.value?.confirmed.country
  if (target) return targetLabel(target)
  if (session.confirmedIso.value) return countryLabel(session.confirmedIso.value)
  if (guided.active.value) {
    return guidedCandidateSurfaceVisible.value ? '国家待确认' : '国家 / 地区'
  }
  return hasSession.value ? '国家待确认' : '国家'
})

const cityBreadcrumb = computed(() => {
  const target = guided.dialogue.value?.confirmed.city
  if (target) return targetLabel(target)
  if (guided.active.value && guided.dialogue.value?.stage === 'city') return '城市待确认'
  return '城市'
})

const liveStatus = computed<LiveStatus>(() => {
  const failed = pendingMessages.value.find((item) => item.status === 'failed')
  if (failed) return { tone: 'warn', text: `发送失败：${failed.error ?? '请重试'} · 可重试或删除` }
  if (guided.conflict.value) return { tone: 'warn', text: safeGuidedText(guided.conflict.value) }
  if (guided.error.value) return { tone: 'warn', text: safeGuidedText(guided.error.value) }
  if (
    (guided.applying.value || guided.loading.value) &&
    guided.pendingOperation.value
  ) {
    return { tone: 'info', text: guided.pendingOperation.value.label }
  }
  if (guided.applying.value) return { tone: 'info', text: '正在处理…' }
  if (guided.loading.value) return { tone: 'info', text: '正在加载…' }
  if (feedbackBusy.value) return { tone: 'info', text: '正在发送…' }
  if (uploading.value) return { tone: 'info', text: '正在分析照片…' }
  if (lastOutcome.value) return lastOutcome.value
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return { tone: 'muted', text: '正在加载…' }
    if (current.stage === 'resolved') return { tone: 'ok', text: `已确认城市：${targetLabel(current.confirmed.city)}` }
    if (!guidedCandidateSurfaceVisible.value) {
      if (guided.reviewReady.value) {
        return { tone: 'ok', text: '核对完成 · 点“下一步”查看候选' }
      }
      const review = guided.reviewProgress.value
      return {
        tone: 'info',
        text: guidedPendingQuestionText.value
          || (review
            ? `核对观察 · 第 ${review.current} / ${review.total} 条`
            : `核对观察 · 还有 ${guided.pendingClueCount.value} 条`),
      }
    }
    if (!guidedConfirmationReady.value) {
      return { tone: 'muted', text: '有新的观察，核对后才能确认地点' }
    }
    if (current.pending_question && !guided.confidenceShortcut.value) {
      return { tone: 'info', text: guidedPendingQuestionText.value || '请先回答上面的问题。' }
    }
    return { tone: 'muted', text: `${guidedStageTitle(current.stage)} · 请选择或排除候选` }
  }
  if (hasSession.value) {
    if (session.confirmedIso.value) {
      const country = session.scores.value.find((item) => item.iso === session.confirmedIso.value)
      const scope = country ? regionScopeLabel(country) : null
      return {
        tone: 'ok',
        text: `已确认：${country ? displayCountryName(country) : session.confirmedIso.value}${scope ? `（${scope}）` : ''}`,
      }
    }
    return { tone: 'muted', text: `候选 ${session.scores.value.length} 个 · 可确认或排除` }
  }
  return { tone: 'muted', text: '上传照片后开始定位' }
})

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

function setError(caught: unknown, fallback: string): void {
  errorMessage.value = normalizeApiError(caught, undefined, fallback).message
}

function releasePreviewUrl(): void {
  if (localPreviewUrl.value?.startsWith('blob:')) URL.revokeObjectURL(localPreviewUrl.value)
  localPreviewUrl.value = null
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
  pointStatus.value = '在照片上点选或框选一处，然后点“分析选定区域”。'
  pointStatusKind.value = 'idle'
  resetGuidedFreeTextSubmission()
  resetObservationAdoption()
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

function openFocusedClue(clue: Clue): void {
  if (!session.imageUrl.value) return
  focusedClue.value = clue
  closeMobileEvidenceForViewer()
  viewerOpen.value = true
}

function openImageViewer(): void {
  if (!session.imageUrl.value) return
  focusedClue.value = null
  closeMobileEvidenceForViewer()
  viewerOpen.value = true
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
  orphanPending('照片已更换，这条消息没有发出')
  lastOutcome.value = null
  restoreDefaultDocks()
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
  orphanPending('照片已更换，这条消息没有发出')
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
  orphanPending('分析新照片期间，这条消息没有发出')
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
    restoreDefaultDocks()
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
        ? '所选国家/地区已被排除，城市列表已清空。'
        : '该国家/地区已不在候选列表中，城市列表已清空。'
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
      cityNotice.value = '该区域已被排除，城市列表已清空。'
      return
    }
    session.setCityRows(kept)
    void requestCities(cityIso, prev.citySourceIso)
    cityNotice.value = `已按新的排除条件筛选（移除 ${before - kept.length} 个），正在刷新…`
    return
  }
  if (cluesChanged || rankChanged) {
    cityNotice.value = '线索已更新，正在刷新城市列表…'
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
    parts.push('已收到反馈')
  }
  const notes: string[] = []
  if (response.done && response.iso) notes.push(`已确认国家：${countryLabel(response.iso)}`)
  const added = session.exclusions.value.filter((code) => !prev.exclusions.includes(code))
  if (added.length) notes.push(`已排除：${added.map((code) => exclusionLabel(code)).join('、')}`)
  if (clueSignature(session.clues.value) !== prev.clueSig) notes.push('线索已更新')
  if (rankingSignature(session.scores.value) !== prev.rankSig) notes.push('候选已更新')
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
      updatePending(pending.id, { status: 'failed', error: '已被新的操作取代，没有发出' })
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
      updatePending(pending.id, { status: 'failed', error: '已被新的操作取代，没有发出' })
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
    updatePending(pending.id, { status: 'failed', error: '请先上传照片' })
    return
  }
  // Never rebind an old-photo bubble to a different session or epoch: the user
  // must resubmit it as a new message instead.
  if (pending.sessionId !== session.sessionId.value || pending.epoch !== sessionEpoch.value) {
    updatePending(pending.id, {
      status: 'failed',
      error: '照片已更换，请重新输入或删除这条消息',
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
      ? { ...record, status: 'failed', error: '已改用新的选区，这次分析没有采用。' }
      : record,
  )
  pointStatus.value = '已选择新区域，点“分析选定区域”开始分析。'
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
    markRegionRequest(requestId, { status: 'failed', error: '照片已更换，请重新选择区域。' })
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
      markRegionRequest(requestId, { status: 'failed', error: '已改用新的选区，这次分析没有采用。' })
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
        ? `分析完成，新增 ${newClueCount} 条线索。`
        : '分析完成，没有新增线索。'
    } else if (report.dedupe === 'duplicate_request' || report.dedupe === 'duplicate_region') {
      completionText = `分析完成：${report.summary || '这处区域已经分析过了。'}`
    } else if (!applied) {
      completionText = `分析完成：${report.summary || '没有新增内容。'}`
    } else {
      completionText = `分析完成：${report.summary || `新增 ${newClueCount} 条线索`}`
    }

    // Evidence mutates in the same session; keep global busy true until the
    // authoritative guided snapshot has been reloaded.
    if (guided.active.value) {
      const reloaded = await guided.reload()
      if (stale()) {
        markRegionRequest(requestId, { status: 'failed', error: '已改用新的选区，这次分析没有采用。' })
        return
      }
      if (!reloaded) {
        const syncError = `区域分析完成，但最新状态加载失败：${guided.error.value ?? '请重试'}`
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
      markRegionRequest(requestId, { status: 'failed', error: '已改用新的选区，这次分析没有采用。' })
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
      cityNotice.value ||= '城市查询暂时不可用，请稍后重试。'
    } else if (response.status === 'no_cities' || !response.city_scores.length) {
      cityNotice.value ||= source
        ? '该区域暂无城市候选，可切换区域范围或稍后重试。'
        : '该国家/地区暂无城市候选，可稍后重试。'
    }
  } catch {
    if (epoch !== sessionEpoch.value || requestNonce !== cityRequestNonce) return
    // Degrade inline: a failed city lookup must never raise a global error banner.
    session.clearCities()
    citySourceLabel.value = null
    cityNotice.value = '城市查询失败，请稍后重试。'
  } finally {
    if (epoch === sessionEpoch.value && requestNonce === cityRequestNonce) cityBusy.value = false
  }
}

const candidatePanelBindings = computed<CandidatePanelBindings>(() => ({
  scores: guidedCountryScores.value,
  confirmedIso: session.confirmedIso.value,
  selectedIso: selectedIso.value,
  exclusions: exclusionChips.value,
  busy: busy.value,
  guidedCandidates: guidedVisibleCandidates.value,
  guidedCandidatesUnlocked: guidedCandidateSurfaceVisible.value,
  guidedConfirmationReady: guidedConfirmationReady.value,
  guidedDataReady: guidedDataReady.value,
  guidedStage: guided.dialogue.value?.stage ?? null,
  cityScores: visibleCityScores.value,
  cityCountryIso: session.cityCountryIso.value,
  citiesStatus: session.citiesStatus.value,
  countryOptions: countryOptions.value,
  cityBusy: cityBusy.value,
  cityNotice: cityNotice.value,
  citySourceIso: citySourceIso.value,
  citySourceLabel: citySourceLabel.value,
}))

const guidedDialogueBindings = computed<GuidedDialogueBindings>(() => ({
  dialogue: guided.dialogue.value,
  candidates: guided.topCandidates.value,
  action: guided.action.value,
  confidence: guided.confidence.value,
  confidenceShortcut: guided.confidenceShortcut.value,
  confidenceCandidates: guided.confidenceCandidates.value,
  loading: guided.loading.value,
  applying: guided.applying.value,
  externalBusy: uploading.value || feedbackBusy.value || pointBusy.value || cityBusy.value,
  error: guided.error.value,
  conflict: guided.conflict.value,
  canRetry: guided.canRetry.value,
  receipt: guided.receipt.value,
  selectedGuidedTargetKey: selectedGuidedTargetKey.value,
  suggestedFreeText: guidedDraftText.value,
  freeTextSubmission: guidedFreeTextSubmission.value,
  phase: guided.phase.value,
  candidatesUnlocked: guidedRankingsUnlocked.value,
  canReviewBack: guided.canReviewBack.value,
  locationConfirmationReady: guidedConfirmationReady.value,
  recentClueIds: guided.recentClueIds.value,
}))
</script>

<template>
  <div class="app-shell immersive-app" :class="{ 'is-mobile-layout': isMobile, 'is-compact-layout': isCompact }">
    <CommandBar
      :root-ref="layoutElementRefs.commandBar"
      :background-inert="backgroundInert"
      :has-session="hasSession"
      :busy="busy"
      :session-status="sessionStatus"
      :live-tone="liveStatus.tone"
      :country-breadcrumb="countryBreadcrumb"
      :city-breadcrumb="cityBreadcrumb"
      :focus-mode="focusMode"
      :history-open="historyOpen"
      :history-allowed="historyAllowed"
      @focus-mode="setFocusMode"
      @open-history="openHistory"
      @new-analysis="newAnalysis"
    />

    <main class="map-stage" aria-label="候选地图" :inert="backgroundInert">
      <div :ref="layoutElementRefs.mapHost" class="map-host">
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

    <EvidenceDock
      :root-ref="layoutElementRefs.evidenceDock"
      :dialog-ref="layoutElementRefs.mobileEvidenceDialog"
      :visible="evidenceAnchorVisible"
      :collapsed="showEvidenceThumbnail"
      :is-mobile="isMobile"
      :mobile-open="mobileEvidenceOpen"
      :inert="historyOpen"
      :image-present="Boolean(session.imageUrl.value)"
      :preview-url="previewUrl"
      :clue-count="clueCount"
      :region-report-count="regionReports.length"
      :click-points="clickPoints"
      :clues="session.clues.value"
      :recent-clue-keys="recentClueKeys"
      :uploading="uploading"
      :upload-notice="uploadNotice"
      :selected-file-name="selectedFile?.name ?? null"
      :has-session="hasSession"
      :has-local-preview="Boolean(localPreviewUrl)"
      :candidates="candidatePanelBindings"
      :guided-active="guided.active.value"
      :guided-country-allowed="!guided.active.value || guided.dialogue.value?.stage === 'country'"
      :mobile-tab="mobileEvidenceTab"
      @expand="expandEvidenceDock"
      @collapse="collapseEvidenceDock"
      @close="closeMobileEvidence"
      @open-tab="openMobileEvidence"
      @file="onFile"
      @upload="upload"
      @open-viewer="openImageViewer"
      @locate-clue="openFocusedClue"
    />

    <ReasoningDock
      :root-ref="layoutElementRefs.reasoningDock"
      :visible="reasoningAnchorVisible"
      :collapsed="showReasoningTaskChip"
      :wide="reasoningWide"
      :is-mobile="isMobile"
      :sheet-state="mobileSheetState"
      :inert="backgroundInert"
      :guided-active="guided.active.value"
      :guided-loading="guided.active.value && !guided.dialogue.value"
      :messages="session.messages.value"
      :pending-messages="pendingMessages"
      :live-status="liveStatus"
      :question-pending="questionPending"
      :inference-notice="inferenceNotice"
      :guided="guidedDialogueBindings"
      :candidates="candidatePanelBindings"
      :feedback-action="session.action.value"
      :feedback-draft="feedbackDraft"
      :feedback-busy="feedbackBusy || uploading"
      :has-session="hasSession"
      :legacy-loading="pointBusy || feedbackBusy"
      @expand="expandReasoningDock"
      @collapse="collapseReasoningDock"
      @toggle-width="toggleReasoningWidth"
      @sheet-state="setMobileSheetState"
      @update:feedback-draft="feedbackDraft = $event"
      @feedback="onFeedback"
      @retry-pending="retryPending"
      @dismiss-pending="dismissPending"
      @confirm="onCountryConfirm"
      @reject="onCountryReject"
      @select-country="selectedIso = $event"
      @select-guided-target="selectGuidedTarget"
      @confirm-guided-target="confirmGuidedTarget"
      @reject-guided-target="rejectGuidedTarget"
      @request-cities="requestCities"
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
      @retry="retryGuidedOperation"
    />

    <HistoryDrawer
      :dialog-ref="layoutElementRefs.historyDialog"
      :open="historyOpen"
      :messages="session.messages.value"
      :pending-messages="pendingMessages"
      :loading="pointBusy || feedbackBusy"
      @close="closeHistory"
      @retry="retryPending"
      @dismiss="dismissPending"
    />

    <ErrorBanner
      :message="errorMessage"
      :inert="backgroundInert"
      @close="errorMessage = null"
    />

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
      :observation-adoption="observationAdoption"
      :observation-adoption-retryable="observationAdoptionRetryable"
      @analyze="analyzeRegionRequest"
      @select-region="selectedRegionReportId = $event"
      @retry-region="retryRegionRequest"
      @ask-question="useRegionQuestion"
      @adopt-observation="onAdoptObservation"
      @retry-observation-adoption="retryObservationAdoption"
      @selection-change="onRegionSelectionChange"
      @close="viewerOpen = false"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '@/api'
import { normalizeApiError } from '@/api/errors'
import type {
  GuidedDialogueResponse,
  Country,
  CountryScore,
  FeedbackPayload,
  FeedbackResponse,
  PendingChatMessage,
} from '@/types'
import { useGeoSession } from '@/composables/useGeoSession'
import { useGuidedDialogue } from '@/composables/useGuidedDialogue'
import { clueValueLabel, countGroupedClues, dimensionLabel } from '@/utils/clues'
import { isQuestionAction } from '@/utils/dialogue'
import { guidedStageTitle, targetLabel } from '@/utils/guided'
import { feedbackReceipt, receiptText, type FeedbackReceiptTone } from '@/utils/feedback'
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
const pointStatus = ref('点击图片任意位置，AI 将分析该区域的文字、颜色、植被等线索。')
const pointStatusKind = ref<'idle' | 'loading' | 'success' | 'error'>('idle')
const viewerOpen = ref(false)
const insightsOpen = ref(true)
const clickPoints = ref<Array<{ x: number; y: number }>>([])
const confirmCandidate = ref<(typeof session.scores.value)[number] | null>(null)
const selectedIso = ref<string | null>(null)
const countryDirectory = ref<Country[]>([])
const feedbackDraft = ref('')
const pendingMessages = ref<PendingChatMessage[]>([])
const lastOutcome = ref<{ tone: FeedbackReceiptTone | 'info'; text: string } | null>(null)
const citySourceIso = ref<string | null>(null)
const citySourceLabel = ref<string | null>(null)
const knownCountryNames = ref<Record<string, string>>({})

/** Monotonic session generation: stale responses can never overwrite newer state. */
const sessionEpoch = ref(0)
/** Monotonic city-request nonce: older city responses must not replace newer country picks. */
let cityRequestNonce = 0
let pendingNonce = 0

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
const questionPending = computed(() =>
  guided.active.value
    ? guided.pendingQuestion.value !== null
    : isQuestionAction(session.action.value),
)
const confirmSubregion = computed(() =>
  confirmCandidate.value ? regionScopeLabel(confirmCandidate.value) : null,
)

/** Region-aware exclusion chips: raw TW/HK/MO never render as 中国 excluded. */
const exclusionChips = computed(() =>
  session.exclusions.value.map((iso) => ({ iso: normalizeIsoCode(iso), label: exclusionLabel(iso) })),
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

const sessionStatus = computed(() => {
  if (uploading.value) return '正在分析图像…'
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return guided.loading.value ? '正在启动引导定位…' : '引导定位待重试'
    if (current.stage === 'resolved') {
      return `已完成：${targetLabel(current.confirmed.city)}`
    }
    return `${guidedStageTitle(current.stage)} · 候选 ${guided.candidates.value.length} 个`
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

const liveStatus = computed<{ tone: FeedbackReceiptTone | 'info'; text: string }>(() => {
  const failed = pendingMessages.value.find((item) => item.status === 'failed')
  if (failed) return { tone: 'warn', text: `发送失败：${failed.error ?? '未知错误'} · 可在对话中重试或删除` }
  if (guided.conflict.value) return { tone: 'warn', text: guided.conflict.value }
  if (guided.error.value) return { tone: 'warn', text: guided.error.value }
  if (guided.applying.value) return { tone: 'info', text: '正在提交引导操作，等待服务器确认…' }
  if (guided.loading.value) return { tone: 'info', text: '正在同步引导状态…' }
  if (feedbackBusy.value) return { tone: 'info', text: '正在提交反馈，等待服务器回执…' }
  if (uploading.value) return { tone: 'info', text: '正在分析图像…' }
  if (lastOutcome.value) return lastOutcome.value
  if (guided.active.value) {
    const current = guided.dialogue.value
    if (!current) return { tone: 'muted', text: '等待引导状态' }
    if (current.stage === 'resolved') return { tone: 'ok', text: `已确认城市：${targetLabel(current.confirmed.city)}` }
    if (current.pending_question) return { tone: 'info', text: current.pending_question.message }
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
  confirmCandidate.value = null
  clickPoints.value = []
  selectedIso.value = null
  cityNotice.value = null
  citySourceIso.value = null
  citySourceLabel.value = null
  pointStatus.value = '点击图片任意位置，AI 将分析该区域的文字、颜色、植被等线索。'
  pointStatusKind.value = 'idle'
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
  insightsOpen.value = true
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
    insightsOpen.value = true
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

/** Chat/feedback buttons requesting confirmation must go through the dialog. */
function routeGuidedFeedback(payload: FeedbackPayload): boolean {
  if (!guided.active.value) return false
  if (guided.loading.value || guided.applying.value) return true
  switch (payload.type) {
    case 'free_text':
      if (payload.text?.trim()) void guided.freeText(payload.text)
      break
    case 'confirm':
      if (payload.iso) void guided.confirmCountry(payload.iso)
      break
    case 'reject':
      if (payload.iso) void guided.rejectCountry(payload.iso)
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
    // Resolve the exact backend target inside the guided composable; never
    // reconstruct a target from the legacy country score.
    void guided.confirmCountry(candidate.iso)
    return
  }
  const payload: FeedbackPayload = { type: 'confirm', iso: candidate.iso }
  void submitFeedback(payload, { content: feedbackContent(payload) })
}

function onCountryConfirm(score: CountryScore): void {
  if (guided.active.value && (guided.loading.value || guided.applying.value)) return
  if (guided.active.value && guided.dialogue.value?.stage !== 'country') {
    void guided.confirmCountry(score.iso)
    return
  }
  confirmCandidate.value = score
}

function onCountryReject(score: CountryScore): void {
  if (guided.active.value && (guided.loading.value || guided.applying.value)) return
  if (guided.active.value) {
    void guided.rejectCountry(score.iso)
    return
  }
  onFeedback({ type: 'reject', iso: score.iso })
}

async function analyzePoint(point: { x: number; y: number }): Promise<void> {
  const id = session.sessionId.value
  if (id === null || pointBusy.value || guided.loading.value || guided.applying.value) return
  const epoch = sessionEpoch.value
  const snapshot = captureCandidateSnapshot()
  pointBusy.value = true
  pointStatus.value = '正在分析该区域…'
  pointStatusKind.value = 'loading'
  errorMessage.value = null
  try {
    const response = await api.analyzePoint(id, point)
    if (epoch !== sessionEpoch.value) return
    session.applyPointAnalysis(response)
    clickPoints.value = [...clickPoints.value, point]
    handleCityStaleness(snapshot)
    const newClueCount = response.new_clues?.length ?? 0

    // Point analysis mutates evidence in the same session. Wait for the
    // authoritative guided snapshot before releasing pointBusy so an older
    // dialogue response cannot re-enable controls against stale evidence.
    if (guided.active.value) {
      const reloaded = await guided.reload()
      if (epoch !== sessionEpoch.value) return
      if (!reloaded) {
        pointStatus.value = `分析完成，发现 ${newClueCount} 条新线索，但引导状态同步失败：${guided.error.value ?? '请重试'}`
        pointStatusKind.value = 'error'
      } else {
        pointStatus.value = `分析完成，发现 ${newClueCount} 条新线索，引导状态已同步。`
        pointStatusKind.value = 'success'
      }
    } else {
      pointStatus.value = `分析完成，发现 ${newClueCount} 条新线索，候选国家评分已更新。`
      pointStatusKind.value = 'success'
    }
  } catch (caught) {
    if (epoch !== sessionEpoch.value) return
    const message = normalizeApiError(caught, undefined, '局部区域分析失败').message
    pointStatus.value = `分析失败：${message}`
    pointStatusKind.value = 'error'
    setError(caught, '局部区域分析失败')
  } finally {
    if (epoch === sessionEpoch.value) pointBusy.value = false
  }
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
  <div class="app-shell">
    <header class="top-bar">
      <div class="brand">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9S14.5 18.4 12 21c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z" />
        </svg>
        地理定位推理台
      </div>
      <div class="sub">候选国家推断 · 参考街景检索 · 人工确认</div>
      <div class="session-status">
        <span class="session-dot" :class="{ active: hasSession, busy: busy }" />
        <span>{{ sessionStatus }}</span>
        <button type="button" class="ghost-btn" @click="newAnalysis">新建分析</button>
      </div>
    </header>

    <main class="layout">
      <aside class="column column-left">
        <section class="panel">
          <div class="panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-4.5-4.5L7 20" />
            </svg>
            图像上传
          </div>
          <UploadPanel
            :preview-url="previewUrl"
            :click-points="clickPoints"
            @file="onFile"
          />
          <div v-if="uploadNotice" class="inline-error">{{ uploadNotice }}</div>
          <div v-if="selectedFile" class="file-chip">
            <span>{{ selectedFile.name }}</span>
            <span class="state">{{ hasSession && !localPreviewUrl ? '已分析' : '待上传' }}</span>
          </div>
          <button
            type="button"
            class="primary-btn upload-btn"
            :disabled="!selectedFile || uploading"
            @click="upload"
          >
            {{ uploading ? '上传并分析中…' : '上传并分析' }}
          </button>
          <button
            v-if="session.imageUrl.value"
            type="button"
            class="link-btn"
            @click="viewerOpen = true"
          >
            打开局部分析查看器
          </button>
        </section>

        <section class="panel clue-panel">
          <div class="panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M4 6h16M4 12h16M4 18h10" />
            </svg>
            线索卡 <span class="count">{{ clueCount }}</span>
          </div>
          <ClueList :clues="session.clues.value" :loading="uploading" />
        </section>
      </aside>

      <section class="workspace">
        <section class="panel chat-panel">
          <div class="chat-head">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z" />
            </svg>
            <span class="chat-title">{{ guided.active.value ? '引导定位' : '人机协同对话' }}</span>
            <span class="panel-count">{{ session.messages.value.length }} 条</span>
            <span v-if="questionPending" class="chat-alert">待回答</span>
            <div class="chat-live" :class="liveStatus.tone" role="status" aria-live="polite">
              <span class="live-dot" aria-hidden="true" />
              <span>{{ liveStatus.text }}</span>
            </div>
          </div>
          <div class="chat-body" :class="{ 'guided-chat-body': guided.active.value }">
            <GuidedDialoguePanel
              v-if="guided.active.value"
              :dialogue="guided.dialogue.value"
              :candidates="guided.candidates.value"
              :loading="guided.loading.value"
              :applying="guided.applying.value"
              :external-busy="uploading || feedbackBusy || pointBusy || cityBusy"
              :error="guided.error.value"
              :conflict="guided.conflict.value"
              :can-retry="guided.canRetry.value"
              :receipt="guided.receipt.value"
              @confirm-location="guided.confirmLocation"
              @reject-location="guided.rejectLocation"
              @confirm-clue="guided.confirmClue"
              @reject-clue="guided.rejectClue"
              @answer="(payload) => guided.answer(payload.value)"
              @unknown="guided.unknown()"
              @retract-target="guided.retractTarget"
              @retract-clue="guided.retractClue"
              @backtrack="guided.backtrack"
              @free-text="guided.freeText"
              @retry="guided.retry"
            />
            <ChatLog
              v-if="!guided.active.value"
              :messages="session.messages.value"
              :pending="pendingMessages"
              :loading="pointBusy || feedbackBusy"
              @retry="retryPending"
              @dismiss="dismissPending"
            />
            <FeedbackBar
              v-if="!guided.active.value"
              v-model:draft="feedbackDraft"
              :action="session.action.value"
              :busy="feedbackBusy || uploading"
              :disabled="!hasSession"
              @feedback="onFeedback"
            />
            <details v-else class="guided-history">
              <summary>查看推理记录（{{ session.messages.value.length }} 条）</summary>
              <ChatLog
                :messages="session.messages.value"
                :pending="pendingMessages"
                :loading="pointBusy || feedbackBusy"
                @retry="retryPending"
                @dismiss="dismissPending"
              />
            </details>
          </div>
        </section>

        <section class="panel insights-panel" :class="{ 'is-collapsed': !insightsOpen }">
          <CandidatePanel
            embedded
            collapsible
            :collapsed="!insightsOpen"
            :scores="session.scores.value"
            :confirmed-iso="session.confirmedIso.value"
            :selected-iso="selectedIso"
            :exclusions="exclusionChips"
            :busy="busy"
            :guided-mode="guided.active.value"
            :guided-country-allowed="!guided.active.value || guided.dialogue.value?.stage === 'country'"
            :city-scores="session.cityScores.value"
            :city-country-iso="session.cityCountryIso.value"
            :cities-status="session.citiesStatus.value"
            :country-options="countryOptions"
            :city-busy="cityBusy"
            :city-notice="cityNotice"
            :city-source-iso="citySourceIso"
            :city-source-label="citySourceLabel"
            @toggle-collapse="insightsOpen = !insightsOpen"
            @expand="insightsOpen = true"
            @confirm="onCountryConfirm"
            @reject="onCountryReject"
            @select-country="selectedIso = $event"
            @request-cities="requestCities"
          >
            <template #map>
              <GeoMap
                :scores="session.scores.value"
                :confirmed-iso="session.confirmedIso.value"
                :city-scores="session.cityScores.value"
                :city-country-iso="session.cityCountryIso.value"
                :selected-iso="selectedIso"
                @confirm="onCountryConfirm"
                @select-country="selectedIso = $event"
              />
            </template>
          </CandidatePanel>
        </section>
      </section>
    </main>

    <transition name="fade">
      <div v-if="errorMessage" class="error-banner" role="alert">
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
      :disabled="!hasSession || guided.loading.value || guided.applying.value"
      @analyze="analyzePoint"
      @close="viewerOpen = false"
    />
  </div>
</template>





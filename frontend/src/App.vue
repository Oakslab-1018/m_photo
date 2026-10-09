<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api } from '@/api'
import { normalizeApiError } from '@/api/errors'
import type { Country, FeedbackPayload } from '@/types'
import { useGeoSession } from '@/composables/useGeoSession'
import UploadPanel from '@/components/UploadPanel.vue'
import ClueList from '@/components/ClueList.vue'
import ChatLog from '@/components/ChatLog.vue'
import GeoMap from '@/components/GeoMap.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import CandidatePanel from '@/components/CandidatePanel.vue'
import ConfirmLocationDialog from '@/components/ConfirmLocationDialog.vue'
import ImageViewer from '@/components/ImageViewer.vue'

const session = useGeoSession()

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
const chatOpen = ref(false)
const clickPoints = ref<Array<{ x: number; y: number }>>([])
const confirmCandidate = ref<(typeof session.scores.value)[number] | null>(null)
const selectedIso = ref<string | null>(null)
const countryDirectory = ref<Country[]>([])

/** Monotonic session generation: stale responses can never overwrite newer state. */
const sessionEpoch = ref(0)
/** Monotonic city-request nonce: older city responses must not replace newer country picks. */
let cityRequestNonce = 0

const busy = computed(() => uploading.value || feedbackBusy.value || pointBusy.value || cityBusy.value)
const hasSession = computed(() => session.sessionId.value !== null)
const previewUrl = computed(() => localPreviewUrl.value ?? session.imageUrl.value)

const sessionStatus = computed(() => {
  if (uploading.value) return '正在分析图像…'
  if (session.confirmedIso.value) {
    const country = session.scores.value.find((item) => item.iso === session.confirmedIso.value)
    return `已由用户确认国家：${country?.name_zh ?? session.confirmedIso.value}`
  }
  if (!hasSession.value) return '未开始 · 请上传一张照片'
  return `候选国家 ${session.scores.value.length} 个 · 均为代表点`
})

const countryOptions = computed(() =>
  countryDirectory.value.length
    ? countryDirectory.value.map((country) => ({ iso: country.iso, name_zh: country.name_zh }))
    : session.scores.value.map((score) => ({ iso: score.iso, name_zh: score.name_zh })),
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
  pointStatus.value = '点击图片任意位置，AI 将分析该区域的文字、颜色、植被等线索。'
  pointStatusKind.value = 'idle'
}

function cancelInFlightUi(): void {
  uploading.value = false
  feedbackBusy.value = false
  pointBusy.value = false
  cityBusy.value = false
}

function newAnalysis(): void {
  sessionEpoch.value += 1
  session.reset()
  cancelInFlightUi()
  releasePreviewUrl()
  selectedFile.value = null
  errorMessage.value = null
  uploadNotice.value = null
  resetTransientUi()
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
  session.reset()
  cancelInFlightUi()
  resetTransientUi()
  releasePreviewUrl()
  selectedFile.value = file
  localPreviewUrl.value = URL.createObjectURL(file)
  uploadNotice.value = null
  errorMessage.value = null
}

async function upload(): Promise<void> {
  if (!selectedFile.value || uploading.value) return
  const epoch = ++sessionEpoch.value
  const file = selectedFile.value
  // A new upload supersedes city/feedback/point requests; their stale finally
  // blocks are epoch-guarded, so clear the shared busy flags here.
  cancelInFlightUi()
  uploading.value = true
  uploadNotice.value = null
  errorMessage.value = null
  try {
    const response = await api.upload(file)
    if (epoch !== sessionEpoch.value) return
    session.applyUpload(response)
    releasePreviewUrl()
    resetTransientUi()
    selectedIso.value = session.scores.value[0]?.iso ?? null
  } catch (caught) {
    if (epoch === sessionEpoch.value) setError(caught, '上传失败，请稍后重试')
  } finally {
    if (epoch === sessionEpoch.value) uploading.value = false
  }
}

async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  const id = session.sessionId.value
  if (id === null || feedbackBusy.value) return
  const epoch = sessionEpoch.value
  feedbackBusy.value = true
  errorMessage.value = null
  try {
    const response = await api.feedback(id, payload)
    if (epoch !== sessionEpoch.value) return
    session.applyFeedback(response)
    if (response.exclusions && session.cityCountryIso.value &&
        response.exclusions.includes(session.cityCountryIso.value)) {
      // The country whose cities are shown was excluded: drop stale city rows.
      session.clearCities()
      cityNotice.value = null
    }
    if (response.done && response.iso) {
      // Confirming one country invalidates city candidates of any other country.
      if (session.cityCountryIso.value && session.cityCountryIso.value !== response.iso) {
        session.clearCities()
        cityNotice.value = null
      }
      selectedIso.value = response.iso
    }
  } catch (caught) {
    if (epoch === sessionEpoch.value) setError(caught, '反馈失败，请稍后重试')
  } finally {
    if (epoch === sessionEpoch.value) feedbackBusy.value = false
  }
}

async function analyzePoint(point: { x: number; y: number }): Promise<void> {
  const id = session.sessionId.value
  if (id === null || pointBusy.value) return
  const epoch = sessionEpoch.value
  pointBusy.value = true
  pointStatus.value = '正在分析该区域…'
  pointStatusKind.value = 'loading'
  errorMessage.value = null
  try {
    const response = await api.analyzePoint(id, point)
    if (epoch !== sessionEpoch.value) return
    session.applyPointAnalysis(response)
    clickPoints.value = [...clickPoints.value, point]
    pointStatus.value = `分析完成，发现 ${response.new_clues?.length ?? 0} 条新线索，候选国家评分已更新。`
    pointStatusKind.value = 'success'
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

async function requestCities(iso: string): Promise<void> {
  const target = (iso ?? '').trim().toUpperCase()
  const id = session.sessionId.value
  if (!target || id === null || cityBusy.value) return
  const epoch = sessionEpoch.value
  const requestNonce = ++cityRequestNonce
  cityBusy.value = true
  cityNotice.value = null
  try {
    const response = await api.cities(id, target)
    if (epoch !== sessionEpoch.value || requestNonce !== cityRequestNonce) return
    session.applyCities(response)
    if (response.status === 'unavailable' || response.status === 'no_cities' || !response.city_scores.length) {
      cityNotice.value = '该国家暂无城市候选；可切换其他国家重试。'
    }
  } catch {
    if (epoch !== sessionEpoch.value || requestNonce !== cityRequestNonce) return
    // Older backends 404/405 here: inline notice only, never a global error banner.
    session.clearCities()
    cityNotice.value = '城市候选服务不可用（后端暂未提供该接口），可稍后重试。'
  } finally {
    if (epoch === sessionEpoch.value && requestNonce === cityRequestNonce) cityBusy.value = false
  }
}

/** Chat/feedback buttons requesting confirmation must go through the dialog. */
function onFeedback(payload: FeedbackPayload): void {
  if (payload.type === 'confirm' && payload.iso) {
    const candidate = session.scores.value.find((item) => item.iso === payload.iso)
    if (candidate) {
      confirmCandidate.value = candidate
      return
    }
  }
  void submitFeedback(payload)
}

function doConfirm(): void {
  const candidate = confirmCandidate.value
  if (!candidate) return
  confirmCandidate.value = null
  void submitFeedback({ type: 'confirm', iso: candidate.iso })
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
      <div class="column column-left">
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
            :disabled="!hasSession"
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
            线索卡 <span class="count">{{ session.clues.value.length }}</span>
          </div>
          <ClueList :clues="session.clues.value" :loading="uploading" />
        </section>
      </div>

      <div class="column column-center">
        <section class="panel map-card">
          <div class="panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z" /><path d="M9 3v15M15 6v15" />
            </svg>
            交互地图 <span class="count">标记为候选代表点，不是照片 GPS</span>
          </div>
          <GeoMap
            :scores="session.scores.value"
            :confirmed-iso="session.confirmedIso.value"
            :city-scores="session.cityScores.value"
            :city-country-iso="session.cityCountryIso.value"
            :selected-iso="selectedIso"
            @confirm="confirmCandidate = $event"
            @select-country="selectedIso = $event"
          />
        </section>

        <section class="panel chat-panel">
          <button type="button" class="chat-toggle" @click="chatOpen = !chatOpen">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z" />
            </svg>
            人机协同对话
            <span class="panel-count">{{ session.messages.value.length }} 条</span>
            <span class="chev" :class="{ open: chatOpen }">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path d="m6 9 6 6 6-6" />
              </svg>
            </span>
          </button>
          <div v-if="chatOpen" class="chat-body">
            <ChatLog :messages="session.messages.value" :loading="pointBusy || feedbackBusy" />
            <FeedbackBar
              :action="session.action.value"
              :busy="feedbackBusy || uploading"
              :disabled="!hasSession"
              @feedback="onFeedback"
            />
          </div>
        </section>
      </div>

      <div class="column column-right">
        <CandidatePanel
          :scores="session.scores.value"
          :confirmed-iso="session.confirmedIso.value"
          :selected-iso="selectedIso"
          :exclusions="session.exclusions.value"
          :busy="busy"
          :city-scores="session.cityScores.value"
          :city-country-iso="session.cityCountryIso.value"
          :cities-status="session.citiesStatus.value"
          :country-options="countryOptions"
          :city-busy="cityBusy"
          :city-notice="cityNotice"
          @confirm="confirmCandidate = $event"
          @reject="(score) => submitFeedback({ type: 'reject', iso: score.iso })"
          @select-country="selectedIso = $event"
          @request-cities="requestCities"
        />
      </div>
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
      :busy="feedbackBusy"
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
      :disabled="!hasSession"
      @analyze="analyzePoint"
      @close="viewerOpen = false"
    />
  </div>
</template>

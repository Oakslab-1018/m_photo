<script setup lang="ts">
import { computed, ref } from 'vue'
import { api } from '@/api'
import { normalizeApiError } from '@/api/errors'
import type { FeedbackPayload } from '@/types'
import { useGeoSession } from '@/composables/useGeoSession'
import UploadPanel from '@/components/UploadPanel.vue'
import ClueList from '@/components/ClueList.vue'
import ChatLog from '@/components/ChatLog.vue'
import GeoMap from '@/components/GeoMap.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import ConfirmLocationDialog from '@/components/ConfirmLocationDialog.vue'
import ImageViewer from '@/components/ImageViewer.vue'

const session = useGeoSession()

const selectedFile = ref<File | null>(null)
const localPreviewUrl = ref<string | null>(null)
const uploading = ref(false)
const feedbackBusy = ref(false)
const pointBusy = ref(false)
const errorMessage = ref<string | null>(null)
const pointStatus = ref('点击图片任意位置，AI 将分析该区域的文字、颜色、植被等线索。')
const pointStatusKind = ref<'idle' | 'loading' | 'success' | 'error'>('idle')
const viewerOpen = ref(false)
const clickPoints = ref<Array<{ x: number; y: number }>>([])
const confirmCandidate = ref<(typeof session.scores.value)[number] | null>(null)

/**
 * Monotonic session generation. Every upload / new-file reset bumps it so a
 * late response from an older session can never overwrite newer state.
 */
const sessionEpoch = ref(0)

const busy = computed(() => uploading.value || feedbackBusy.value || pointBusy.value)
const hasSession = computed(() => session.sessionId.value !== null)
const previewUrl = computed(() => localPreviewUrl.value ?? session.imageUrl.value)

function setError(caught: unknown, fallback: string): void {
  errorMessage.value = normalizeApiError(caught, undefined, fallback).message
}

function resetTransientUi(): void {
  viewerOpen.value = false
  confirmCandidate.value = null
  clickPoints.value = []
  pointStatus.value = '点击图片任意位置，AI 将分析该区域的文字、颜色、植被等线索。'
  pointStatusKind.value = 'idle'
}

function onFile(file: File): void {
  // Guard against switching files while a request is in flight: a stale
  // session response would otherwise be applied to the new preview.
  if (busy.value) return
  sessionEpoch.value += 1
  session.reset()
  resetTransientUi()
  if (localPreviewUrl.value?.startsWith('blob:')) URL.revokeObjectURL(localPreviewUrl.value)
  selectedFile.value = file
  localPreviewUrl.value = URL.createObjectURL(file)
  errorMessage.value = null
}

async function upload(): Promise<void> {
  if (!selectedFile.value || busy.value) return
  const epoch = ++sessionEpoch.value
  const file = selectedFile.value
  uploading.value = true
  errorMessage.value = null
  try {
    const response = await api.upload(file)
    if (epoch !== sessionEpoch.value) return
    session.applyUpload(response)
    if (localPreviewUrl.value?.startsWith('blob:')) URL.revokeObjectURL(localPreviewUrl.value)
    localPreviewUrl.value = null
    resetTransientUi()
  } catch (caught) {
    if (epoch === sessionEpoch.value) setError(caught, '上传失败，请稍后重试')
  } finally {
    if (epoch === sessionEpoch.value) uploading.value = false
  }
}

async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  const id = session.sessionId.value
  if (id === null || busy.value) return
  const epoch = sessionEpoch.value
  feedbackBusy.value = true
  errorMessage.value = null
  try {
    const response = await api.feedback(id, payload)
    if (epoch !== sessionEpoch.value) return
    session.applyFeedback(response)
  } catch (caught) {
    if (epoch === sessionEpoch.value) setError(caught, '反馈失败，请稍后重试')
  } finally {
    if (epoch === sessionEpoch.value) feedbackBusy.value = false
  }
}

async function analyzePoint(point: { x: number; y: number }): Promise<void> {
  const id = session.sessionId.value
  if (id === null || busy.value) return
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
      <div class="brand">🌍 AI 地理定位推理系统</div>
      <div class="sub">Vue 3 前端 · 多模态线索提取 · 人机协同验证</div>
      <div class="mode-badge" title="当前为第一阶段独立前端；旧版 Jinja 页面仍由后端保留">阶段一 · 独立前端</div>
    </header>

    <main class="layout">
      <aside class="panel left-panel">
        <div class="panel-title">📷 图像上传</div>
        <UploadPanel
          :preview-url="previewUrl"
          :disabled="busy"
          :click-points="clickPoints"
          @file="onFile"
        />
        <button
          type="button"
          class="primary-btn upload-btn"
          :disabled="!selectedFile || busy"
          @click="upload"
        >
          {{ uploading ? '上传并分析中…' : hasSession ? '重新上传分析' : '上传图片' }}
        </button>
        <button
          v-if="session.imageUrl.value"
          type="button"
          class="link-btn"
          :disabled="!hasSession || busy"
          @click="viewerOpen = true"
        >
          🔍 打开局部分析查看器
        </button>

        <div class="panel-title clue-title">🔍 线索卡</div>
        <ClueList :clues="session.clues.value" :loading="uploading" />
      </aside>

      <section class="panel map-panel">
        <div class="panel-title">🗺️ 推理地图（候选国家高亮）</div>
        <GeoMap
          :scores="session.scores.value"
          :confirmed-iso="session.confirmedIso.value"
          :exclusions="session.exclusions.value"
          :busy="busy"
          @confirm="confirmCandidate = $event"
          @reject="(score) => submitFeedback({ type: 'reject', iso: score.iso })"
        />

        <div class="chat-section">
          <div class="panel-title chat-title">💬 人机协同对话</div>
          <ChatLog :messages="session.messages.value" :loading="pointBusy || feedbackBusy" />
          <FeedbackBar
            :action="session.action.value"
            :busy="feedbackBusy || uploading"
            :disabled="!hasSession || busy"
            @feedback="submitFeedback"
          />
        </div>
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
      :disabled="!hasSession || busy"
      @analyze="analyzePoint"
      @close="viewerOpen = false"
    />
  </div>
</template>

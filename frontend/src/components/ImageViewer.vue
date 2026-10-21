<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type {
  AnalyzePointRequest,
  AnalyzePurpose,
  Clue,
  ImageRegion,
  Point,
  RegionAnalysisRecord,
  RegionQuestion,
  RegionReport,
} from '@/types'
import RegionReportPanel from '@/components/RegionReportPanel.vue'
import { clueRegionBox } from '@/utils/clueRegion'
import { clueSourceLabel, clueValueLabel } from '@/utils/clues'
import { newRegionRequestId, observationBox } from '@/utils/regionAnalysis'

const props = withDefaults(
  defineProps<{
    imageUrl: string | null
    clickPoints?: Point[]
    regionReports?: RegionReport[]
    selectedRegionReportId?: string | null
    regionRequests?: RegionAnalysisRecord[]
    analyzing: boolean
    status: string
    statusKind: 'idle' | 'loading' | 'success' | 'error'
    disabled?: boolean
    focusedClue?: Clue | null
  }>(),
  {
    clickPoints: () => [],
    regionReports: () => [],
    selectedRegionReportId: null,
    regionRequests: () => [],
    disabled: false,
    focusedClue: null,
  },
)

const emit = defineEmits<{
  analyze: [request: AnalyzePointRequest]
  'select-region': [reportId: string]
  'retry-region': [request: AnalyzePointRequest]
  'ask-question': [question: RegionQuestion]
  'selection-change': []
  close: []
}>()

const zoom = ref(1)
const mode = ref<'point' | 'rect' | 'pan'>('point')
const scale = ref(0.25)
const purpose = ref<AnalyzePurpose>('auto')
const draftPoint = ref<Point | null>(null)
const draftRegion = ref<ImageRegion | null>(null)
const imageRef = ref<HTMLImageElement | null>(null)
const wrapRef = ref<HTMLElement | null>(null)
const panning = ref(false)
const rectDragging = ref(false)
let moved = false
let rectOrigin: Point | null = null
let panStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 }

const interactionDisabled = computed(() => props.disabled || props.analyzing)
const selectedReport = computed(() =>
  props.regionReports.find((report) => report.id === props.selectedRegionReportId)
  ?? props.regionReports[0]
  ?? null,
)
const canAnalyze = computed(() => !interactionDisabled.value && buildRequest(false) !== null)
const selectionText = computed(() => {
  if (draftRegion.value && draftRegion.value.w > 0 && draftRegion.value.h > 0) {
    const region = draftRegion.value
    return '矩形 ROI：' + formatPercent(region.x) + ', ' + formatPercent(region.y) + ' · ' + formatPercent(region.w) + ' × ' + formatPercent(region.h)
  }
  if (draftPoint.value) return '点选：' + formatPercent(draftPoint.value.x) + ', ' + formatPercent(draftPoint.value.y)
  return '尚未选择区域'
})
const viewerTip = computed(() => {
  if (interactionDisabled.value) return '分析中或当前忙碌，暂时锁定选区与引导操作'
  if (mode.value === 'rect') return '框选模式：拖拽画出矩形 ROI；滚轮缩放，切换平移动态图'
  if (mode.value === 'pan') return '平移模式：拖拽移动图片；滚轮缩放'
  return '点选模式：单击图片选择点；拖拽平移，滚轮缩放；确认后点击“分析选定区域”'
})
const reportBox = computed(() => selectedReport.value?.bbox_norm ?? null)
const focusedBox = computed(() => clueRegionBox(props.focusedClue))
const focusedClueLabel = computed(() => {
  const source = props.focusedClue ? clueSourceLabel(props.focusedClue) : ''
  return source ? source + ' · 来源区域' : '线索来源区域'
})
const focusedClueTitle = computed(() => {
  if (!props.focusedClue) return ''
  const value = clueValueLabel(props.focusedClue.value)
  const source = clueSourceLabel(props.focusedClue)
  return source ? '线索：' + value + '（' + source + '）' : '线索：' + value
})
const evidenceBoxes = computed(() => {
  const report = selectedReport.value
  if (!report) return []
  return report.observations
    .filter((observation) => observationBox(observation) !== null)
    .filter((observation) => observation.kind !== 'ocr_text' || (report.ocr.available && report.ocr.state.toLowerCase() !== 'mock' && Boolean(observation.text?.trim())))
    .map((observation) => ({ id: observation.id, bbox: observationBox(observation)!, kind: observation.kind }))
})

function formatPercent(value: number): string {
  return Math.round(value * 100) + '%'
}

function boxStyle(bbox: { x: number; y: number; w: number; h: number } | null): Record<string, string> {
  if (!bbox) return {}
  return {
    left: String(bbox.x * 100) + '%',
    top: String(bbox.y * 100) + '%',
    width: String(bbox.w * 100) + '%',
    height: String(bbox.h * 100) + '%',
  }
}

function clampScroll(value: number, maximum: number): number {
  return Math.min(Math.max(0, value), Math.max(0, maximum))
}

/**
 * Center the backend-provided clue source box in the scroll viewport. This is
 * display-only: it never changes the draft selection or starts an analysis.
 */
function scrollToFocusedClue(): void {
  const wrap = wrapRef.value
  const image = imageRef.value
  const bbox = focusedBox.value
  if (!wrap || !image || !bbox || !image.complete || image.naturalWidth <= 0) return
  const wrapRect = wrap.getBoundingClientRect()
  const imageRect = image.getBoundingClientRect()
  if (wrapRect.width <= 0 || wrapRect.height <= 0 || imageRect.width <= 0 || imageRect.height <= 0) return

  const boxLeft = imageRect.left + bbox.x * imageRect.width
  const boxTop = imageRect.top + bbox.y * imageRect.height
  const boxWidth = bbox.w * imageRect.width
  const boxHeight = bbox.h * imageRect.height
  const horizontalTarget = boxWidth <= wrap.clientWidth
    ? boxLeft + boxWidth / 2 - (wrapRect.left + wrap.clientWidth / 2)
    : boxLeft - wrapRect.left - 8
  const verticalTarget = boxHeight <= wrap.clientHeight
    ? boxTop + boxHeight / 2 - (wrapRect.top + wrap.clientHeight / 2)
    : boxTop - wrapRect.top - 8

  wrap.scrollLeft = clampScroll(wrap.scrollLeft + horizontalTarget, wrap.scrollWidth - wrap.clientWidth)
  wrap.scrollTop = clampScroll(wrap.scrollTop + verticalTarget, wrap.scrollHeight - wrap.clientHeight)
}

async function onImageLoad(): Promise<void> {
  await nextTick()
  scrollToFocusedClue()
}

function pointFromEvent(event: MouseEvent): Point | null {
  const image = imageRef.value
  if (!image) return null
  const rect = image.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return null
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  }
}
function buildRequest(assignId: boolean): AnalyzePointRequest | null {
  const region = draftRegion.value
  const point = draftPoint.value
  if (region && region.w > 0 && region.h > 0) {
    const request: AnalyzePointRequest = {
      x: region.x + region.w / 2,
      y: region.y + region.h / 2,
      scale: Math.min(0.8, Math.max(0.08, scale.value)),
      image_region: { ...region },
      purpose: purpose.value,
    }
    if (assignId) request.client_request_id = newRegionRequestId()
    return request
  }
  if (!point) return null
  const request: AnalyzePointRequest = {
    x: point.x,
    y: point.y,
    scale: Math.min(0.8, Math.max(0.08, scale.value)),
    purpose: purpose.value,
  }
  if (assignId) request.client_request_id = newRegionRequestId()
  return request
}

function analyzeSelection(): void {
  if (interactionDisabled.value) return
  const request = buildRequest(true)
  if (request) emit('analyze', request)
}

function clearSelection(): void {
  draftPoint.value = null
  draftRegion.value = null
  rectOrigin = null
}

function setMode(next: 'point' | 'rect' | 'pan'): void {
  if (interactionDisabled.value || mode.value === next) return
  const hadSelection = draftPoint.value !== null || draftRegion.value !== null
  mode.value = next
  if (next === 'point') draftRegion.value = null
  if (next === 'rect') draftPoint.value = null
  if (hadSelection) emit('selection-change')
}

function regionFromPoints(start: Point, end: Point): ImageRegion {
  const x = Math.min(start.x, end.x)
  const y = Math.min(start.y, end.y)
  return {
    x,
    y,
    w: Math.abs(end.x - start.x),
    h: Math.abs(end.y - start.y),
  }
}

function onMouseDown(event: MouseEvent): void {
  if (event.button !== 0 || interactionDisabled.value || !imageRef.value || !wrapRef.value) return
  moved = false
  if (mode.value === 'rect') {
    const point = pointFromEvent(event)
    if (!point) return
    rectOrigin = point
    rectDragging.value = true
    draftRegion.value = { x: point.x, y: point.y, w: 0, h: 0 }
    event.preventDefault()
    return
  }
  panning.value = true
  panStart = {
    x: event.clientX,
    y: event.clientY,
    scrollLeft: wrapRef.value.scrollLeft,
    scrollTop: wrapRef.value.scrollTop,
  }
  event.preventDefault()
}

function onMouseMove(event: MouseEvent): void {
  if (rectDragging.value && rectOrigin) {
    const current = pointFromEvent(event)
    if (!current) return
    if (Math.abs(current.x - rectOrigin.x) > 0.004 || Math.abs(current.y - rectOrigin.y) > 0.004) moved = true
    draftRegion.value = regionFromPoints(rectOrigin, current)
    return
  }
  if (!panning.value || !wrapRef.value) return
  const dx = event.clientX - panStart.x
  const dy = event.clientY - panStart.y
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true
  wrapRef.value.scrollLeft = panStart.scrollLeft - dx
  wrapRef.value.scrollTop = panStart.scrollTop - dy
}

function onMouseUp(): void {
  if (rectDragging.value) {
    rectDragging.value = false
    const region = draftRegion.value
    if (!region || region.w < 0.005 || region.h < 0.005) {
      draftRegion.value = null
    } else {
      emit('selection-change')
    }
    rectOrigin = null
  }
  panning.value = false
}

function onImageClick(event: MouseEvent): void {
  if (moved || interactionDisabled.value || mode.value !== 'point') return
  const point = pointFromEvent(event)
  if (!point) return
  draftPoint.value = point
  draftRegion.value = null
  emit('selection-change')
}

function onWheel(event: WheelEvent): void {
  event.preventDefault()
  const next = event.deltaY < 0 ? zoom.value * 1.15 : zoom.value / 1.15
  zoom.value = Math.min(5, Math.max(1, next))
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') emit('close')
}

function close(): void {
  emit('close')
}

function retryRequest(request: AnalyzePointRequest): void {
  if (!interactionDisabled.value) emit('retry-region', request)
}

function askQuestion(question: RegionQuestion): void {
  if (!interactionDisabled.value) emit('ask-question', question)
}

function selectReport(reportId: string): void {
  emit('select-region', reportId)
}

watch(
  () => props.imageUrl,
  () => {
    zoom.value = 1
    mode.value = 'point'
    scale.value = 0.25
    purpose.value = 'auto'
    clearSelection()
  },
)

watch(
  focusedBox,
  async () => {
    await nextTick()
    scrollToFocusedClue()
  },
  { flush: 'post' },
)

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
})
</script>

<template>
  <div class="image-viewer" role="dialog" aria-modal="true" aria-label="图片局部分析">
    <div class="viewer-backdrop" @click="close" />
    <div class="viewer-content" :class="{ 'with-report': regionReports.length || regionRequests.length }">
      <div class="viewer-header">
        <div class="viewer-heading">
          <span class="viewer-tip">{{ viewerTip }}</span>
          <span v-if="analyzing" class="viewer-busy" role="status">分析中，暂不允许改选区或引导操作</span>
        </div>
        <div class="viewer-tools">
          <div class="viewer-mode-group" role="group" aria-label="图片交互模式">
            <button type="button" class="viewer-mode" :class="{ active: mode === 'point' }" :disabled="interactionDisabled" @click="setMode('point')">
              点选
            </button>
            <button type="button" class="viewer-mode" :class="{ active: mode === 'rect' }" :disabled="interactionDisabled" @click="setMode('rect')">
              框选
            </button>
            <button type="button" class="viewer-mode" :class="{ active: mode === 'pan' }" :disabled="interactionDisabled" @click="setMode('pan')">
              平移
            </button>
          </div>
          <button type="button" class="viewer-zoom" aria-label="缩小" :disabled="interactionDisabled" @click="zoom = Math.max(1, zoom / 1.15)">−</button>
          <span class="viewer-zoom-label">{{ Math.round(zoom * 100) }}%</span>
          <button type="button" class="viewer-zoom" aria-label="放大" :disabled="interactionDisabled" @click="zoom = Math.min(5, zoom * 1.15)">＋</button>
          <button type="button" class="viewer-close" aria-label="关闭" @click="close">×</button>
        </div>
      </div>

      <div class="viewer-controls">
        <label class="viewer-scale-control">
          <span>点击尺度</span>
          <input v-model.number="scale" type="range" min="0.08" max="0.8" step="0.01" :disabled="interactionDisabled">
          <strong>{{ Math.round(scale * 100) }}%</strong>
        </label>
        <div class="viewer-purpose-group" role="group" aria-label="分析用途">
          <button type="button" class="viewer-purpose" :class="{ active: purpose === 'auto' }" :disabled="interactionDisabled" @click="purpose = 'auto'">自动</button>
          <button type="button" class="viewer-purpose" :class="{ active: purpose === 'ocr' }" :disabled="interactionDisabled" @click="purpose = 'ocr'">文字</button>
          <button type="button" class="viewer-purpose" :class="{ active: purpose === 'scene' }" :disabled="interactionDisabled" @click="purpose = 'scene'">场景</button>
        </div>
        <span class="viewer-selection-summary">{{ selectionText }}</span>
        <button type="button" class="primary-btn viewer-analyze-btn" :disabled="!canAnalyze" @click="analyzeSelection">
          {{ analyzing ? '分析中…' : '分析选定区域' }}
        </button>
      </div>

      <div class="viewer-layout">
        <div class="viewer-main">
          <div
            ref="wrapRef"
            class="viewer-image-wrap"
            :class="[`mode-${mode}`, { dragging: panning || rectDragging, disabled: interactionDisabled }]"
            @wheel="onWheel"
          >
            <div class="viewer-stage" :style="{ width: zoom * 100 + '%' }">
              <img
                ref="imageRef"
                :src="imageUrl ?? undefined"
                alt="上传图片大图"
                @load="onImageLoad"
                @mousedown="onMouseDown"
                @click="onImageClick"
              >
              <div class="viewer-markers">
                <span
                  v-for="(point, index) in clickPoints"
                  :key="`point-${point.x}-${point.y}-${index}`"
                  class="click-marker"
                  :style="{ left: point.x * 100 + '%', top: point.y * 100 + '%' }"
                >
                  <span class="click-num">{{ index + 1 }}</span>
                </span>
                <span
                  v-if="draftPoint"
                  class="click-marker draft-point"
                  :style="{ left: draftPoint.x * 100 + '%', top: draftPoint.y * 100 + '%' }"
                >
                  <span class="click-num">选</span>
                </span>
                <span
                  v-if="draftRegion"
                  class="draft-region"
                  :style="boxStyle(draftRegion)"
                >ROI</span>
                <span
                  v-if="reportBox"
                  class="report-region-box"
                  :style="boxStyle(reportBox)"
                >报告 ROI</span>
                <span
                  v-for="box in evidenceBoxes"
                  :key="`evidence-${box.id}`"
                  class="report-evidence-box"
                  :class="`kind-${box.kind}`"
                  :style="boxStyle(box.bbox)"
                />
                <span
                  v-if="focusedBox"
                  class="focused-clue-box"
                  :class="{
                    'label-inside': focusedBox.y < 0.04,
                    'label-right': focusedBox.x + focusedBox.w > 0.75,
                  }"
                  :style="boxStyle(focusedBox)"
                  :title="focusedClueTitle"
                  :aria-label="focusedClueLabel"
                  role="img"
                >
                  <span class="focused-clue-label">{{ focusedClueLabel }}</span>
                </span>
              </div>
            </div>
          </div>
          <div class="viewer-status" :class="statusKind">
            {{ status }}
          </div>
        </div>

        <RegionReportPanel
          v-if="regionReports.length || regionRequests.length"
          :image-url="imageUrl"
          :reports="regionReports"
          :selected-report-id="selectedRegionReportId"
          :requests="regionRequests"
          :analyzing="analyzing"
          :disabled="disabled"
          @select-region="selectReport"
          @retry-region="retryRequest"
          @ask-question="askQuestion"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.focused-clue-box {
  position: absolute;
  z-index: 3;
  box-sizing: border-box;
  border: 2px solid var(--amber);
  background: rgba(232, 163, 61, 0.14);
  box-shadow: 0 0 0 1px rgba(10, 15, 18, 0.65), 0 0 20px rgba(232, 163, 61, 0.34);
  pointer-events: none;
  animation: clue-source-pulse 1.2s ease-out 1;
}

.focused-clue-label {
  position: absolute;
  left: 2px;
  bottom: calc(100% + 2px);
  max-width: min(240px, 70vw);
  overflow: hidden;
  padding: 2px 6px;
  border: 1px solid var(--amber-line);
  border-radius: 3px;
  background: rgba(10, 15, 18, 0.9);
  color: var(--amber);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  text-overflow: ellipsis;
  white-space: nowrap;
  backdrop-filter: blur(6px);
}

.focused-clue-box.label-inside .focused-clue-label {
  top: 2px;
  bottom: auto;
}

.focused-clue-box.label-right .focused-clue-label {
  right: 2px;
  left: auto;
}

@keyframes clue-source-pulse {
  0% {
    box-shadow: 0 0 0 1px rgba(10, 15, 18, 0.65), 0 0 0 7px rgba(232, 163, 61, 0.28);
  }
  100% {
    box-shadow: 0 0 0 1px rgba(10, 15, 18, 0.65), 0 0 20px rgba(232, 163, 61, 0.34);
  }
}

@media (prefers-reduced-motion: reduce) {
  .focused-clue-box {
    animation: none;
  }
}
</style>

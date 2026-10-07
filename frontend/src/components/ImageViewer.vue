<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import type { Point } from '@/types'

const props = defineProps<{
  imageUrl: string | null
  clickPoints: Point[]
  analyzing: boolean
  status: string
  statusKind: 'idle' | 'loading' | 'success' | 'error'
  disabled?: boolean
}>()

const emit = defineEmits<{
  analyze: [point: Point]
  close: []
}>()

const zoom = ref(1)
const imageRef = ref<HTMLImageElement | null>(null)
const wrapRef = ref<HTMLElement | null>(null)
const dragging = ref(false)
let dragStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 }
let moved = false

watch(
  () => props.imageUrl,
  () => {
    zoom.value = 1
  },
)

function close(): void {
  emit('close')
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') close()
}

function onWheel(event: WheelEvent): void {
  event.preventDefault()
  const next = event.deltaY < 0 ? zoom.value * 1.15 : zoom.value / 1.15
  zoom.value = Math.min(5, Math.max(1, next))
}

function onMouseDown(event: MouseEvent): void {
  if (event.button !== 0 || !wrapRef.value) return
  dragging.value = true
  moved = false
  dragStart = {
    x: event.clientX,
    y: event.clientY,
    scrollLeft: wrapRef.value.scrollLeft,
    scrollTop: wrapRef.value.scrollTop,
  }
  event.preventDefault()
}

function onImageClick(event: MouseEvent): void {
  if (moved || props.analyzing || props.disabled || !imageRef.value) return
  const rect = imageRef.value.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return
  const x = (event.clientX - rect.left) / rect.width
  const y = (event.clientY - rect.top) / rect.height
  if (x < 0 || x > 1 || y < 0 || y > 1) return
  emit('analyze', { x, y })
}

function onMouseMove(event: MouseEvent): void {
  if (!dragging.value || !wrapRef.value) return
  const dx = event.clientX - dragStart.x
  const dy = event.clientY - dragStart.y
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true
  wrapRef.value.scrollLeft = dragStart.scrollLeft - dx
  wrapRef.value.scrollTop = dragStart.scrollTop - dy
}

function endDrag(): void {
  dragging.value = false
}

window.addEventListener('keydown', onKeydown)
window.addEventListener('mousemove', onMouseMove)
window.addEventListener('mouseup', endDrag)

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', endDrag)
})
</script>

<template>
  <div class="image-viewer" role="dialog" aria-modal="true" aria-label="图片局部分析">
    <div class="viewer-backdrop" @click="close" />
    <div class="viewer-content">
      <div class="viewer-header">
        <span class="viewer-tip">点击图片任意位置，AI 将分析该区域；滚轮缩放，拖拽平移</span>
        <div class="viewer-tools">
          <button type="button" class="viewer-zoom" aria-label="缩小" @click="zoom = Math.max(1, zoom / 1.15)">−</button>
          <span class="viewer-zoom-label">{{ Math.round(zoom * 100) }}%</span>
          <button type="button" class="viewer-zoom" aria-label="放大" @click="zoom = Math.min(5, zoom * 1.15)">＋</button>
          <button type="button" class="viewer-close" aria-label="关闭" @click="close">×</button>
        </div>
      </div>
      <div
        ref="wrapRef"
        class="viewer-image-wrap"
        :class="{ dragging }"
        @wheel="onWheel"
        @mousedown="onMouseDown"
      >
        <div class="viewer-stage" :style="{ width: `${zoom * 100}%` }">
          <img ref="imageRef" :src="imageUrl ?? undefined" alt="上传图片大图" @click="onImageClick">
          <div class="viewer-markers">
            <span
              v-for="(point, index) in clickPoints"
              :key="`${point.x}-${point.y}-${index}`"
              class="click-marker"
              :style="{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }"
            >
              <span class="click-num">{{ index + 1 }}</span>
            </span>
          </div>
        </div>
      </div>
      <div class="viewer-status" :class="statusKind">{{ status }}</div>
    </div>
  </div>
</template>

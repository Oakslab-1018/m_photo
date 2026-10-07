<template>
  <div
    class="upload-area"
    :class="{ dragover }"
    role="button"
    tabindex="0"
    aria-label="选择或拖拽图片"
    @click="pick"
    @keydown.enter="pick"
    @keydown.space.prevent="pick"
    @dragover.prevent="dragover = true"
    @dragleave="dragover = false"
    @drop.prevent="onDrop"
  >
    <input
      ref="inputRef"
      class="file-input"
      type="file"
      accept="image/*"
      :disabled="disabled"
      @change="onChange"
    >
    <template v-if="previewUrl">
      <img :src="previewUrl" alt="待上传图片预览" class="preview-img">
      <div v-if="clickPoints.length" class="preview-markers">
        <span
          v-for="(point, index) in clickPoints"
          :key="`${point.x}-${point.y}-${index}`"
          class="click-marker"
          :style="{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }"
        >
          <span class="click-num">{{ index + 1 }}</span>
        </span>
      </div>
    </template>
    <div v-else class="upload-hint">
      点击或拖拽图片到此处
      <span>支持 jpg / png / webp</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Point } from '@/types'

const props = defineProps<{
  previewUrl: string | null
  disabled?: boolean
  clickPoints: Point[]
}>()

const emit = defineEmits<{
  file: [file: File]
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const dragover = ref(false)

function pick(): void {
  if (!props.disabled) inputRef.value?.click()
}

function onChange(event: Event): void {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('file', file)
  input.value = ''
}

function onDrop(event: DragEvent): void {
  dragover.value = false
  if (props.disabled) return
  const file = event.dataTransfer?.files?.[0]
  if (file) emit('file', file)
}
</script>

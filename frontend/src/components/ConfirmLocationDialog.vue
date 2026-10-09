<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import type { CountryScore } from '@/types'
import { scoreLabel } from '@/utils/map'

const props = defineProps<{
  country: CountryScore
  busy?: boolean
}>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const okRef = ref<HTMLButtonElement | null>(null)
let previouslyFocused: HTMLElement | null = null

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  event.preventDefault()
  emit('cancel')
}

onMounted(() => {
  previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
  window.addEventListener('keydown', onKeydown)
  okRef.value?.focus()
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  previouslyFocused?.focus?.()
})
</script>

<template>
  <div class="confirm-modal" role="dialog" aria-modal="true" aria-label="确认国家">
    <div class="confirm-backdrop" @click="emit('cancel')" />
    <div class="confirm-box">
      <div class="confirm-msg">
        是否确认这张照片拍摄的国家是「{{ country.name_zh }}」？
        <div class="confirm-sub">
          当前{{ scoreLabel(country) }} · 确认的是候选国家，不是照片 GPS 坐标
        </div>
      </div>
      <div class="confirm-actions">
        <button type="button" class="confirm-cancel" :disabled="busy" @click="emit('cancel')">取消</button>
        <button ref="okRef" type="button" class="confirm-ok" :disabled="busy" @click="emit('confirm')">{{ busy ? '提交中…' : '确认' }}</button>
      </div>
    </div>
  </div>
</template>

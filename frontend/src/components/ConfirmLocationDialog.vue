<script setup lang="ts">
import type { CountryScore } from '@/types'
import { scorePercent } from '@/utils/map'

const props = defineProps<{
  country: CountryScore
  busy?: boolean
}>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()
</script>

<template>
  <div class="confirm-modal" role="dialog" aria-modal="true" aria-label="确认拍摄地">
    <div class="confirm-backdrop" @click="emit('cancel')" />
    <div class="confirm-box">
      <div class="confirm-msg">
        是否将「{{ country.name_zh }}」设置为照片拍摄地？
        <div class="confirm-sub">当前匹配度 {{ scorePercent(country.score) }}</div>
      </div>
      <div class="confirm-actions">
        <button type="button" class="confirm-cancel" :disabled="busy" @click="emit('cancel')">取消</button>
        <button type="button" class="confirm-ok" :disabled="busy" @click="emit('confirm')">{{ busy ? '提交中…' : '确认' }}</button>
      </div>
    </div>
  </div>
</template>

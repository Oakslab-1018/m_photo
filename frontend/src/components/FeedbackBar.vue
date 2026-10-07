<script setup lang="ts">
import { nextTick, ref } from 'vue'
import type { FeedbackPayload, HypothesisAction } from '@/types'

const props = defineProps<{
  action: HypothesisAction | null
  busy: boolean
  disabled?: boolean
}>()

const emit = defineEmits<{
  feedback: [payload: FeedbackPayload]
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const text = ref('')

async function focusInput(): Promise<void> {
  await nextTick()
  inputRef.value?.focus()
}

function submit(): void {
  const value = text.value.trim()
  if (!value || props.busy || props.disabled) return
  emit('feedback', { type: 'free_text', text: value })
  text.value = ''
}

function confirmIso(action: HypothesisAction): void {
  const iso = action.payload?.iso
  if (iso) emit('feedback', { type: 'confirm', iso })
}

function rejectIso(action: HypothesisAction): void {
  const iso = action.payload?.iso
  if (iso) emit('feedback', { type: 'reject', iso })
}

defineExpose({ focusInput })
</script>

<template>
  <div class="feedback-area">
    <div v-if="action" class="action-buttons">
      <button
        v-if="action.kind === 'confirm' && action.payload?.iso"
        type="button"
        class="action-btn success"
        :disabled="busy"
        @click="confirmIso(action)"
      >
        ✅ 确认是这里
      </button>
      <button
        v-if="action.kind === 'confirm' && action.payload?.iso"
        type="button"
        class="action-btn danger"
        :disabled="busy"
        @click="rejectIso(action)"
      >
        ❌ 不是这里
      </button>
      <button
        v-if="action.kind === 'ask' || action.kind === 'request_clue'"
        type="button"
        class="action-btn"
        @click="focusInput"
      >
        ✍️ 我来补充{{ action.payload?.dimensions?.length ? `（${action.payload.dimensions.join('、')}）` : '' }}
      </button>
      <button v-if="action.kind === 'done'" type="button" class="action-btn success" disabled>🎉 推理完成</button>
    </div>
    <div class="chat-input-row">
      <input
        ref="inputRef"
        v-model="text"
        class="chat-input"
        type="text"
        :placeholder="action?.kind === 'ask' || action?.kind === 'request_clue' ? '输入补充线索或直接描述场景…' : '输入反馈，或直接描述线索…'"
        :disabled="disabled || busy"
        @keydown.enter.prevent="submit"
      >
      <button type="button" class="primary-btn send-btn" :disabled="disabled || busy" @click="submit">
        {{ busy ? '处理中…' : '发送' }}
      </button>
    </div>
  </div>
</template>

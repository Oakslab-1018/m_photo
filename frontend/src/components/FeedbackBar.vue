<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
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
const composing = ref(false)

async function focusInput(): Promise<void> {
  await nextTick()
  inputRef.value?.focus()
}

const canSend = computed(() => text.value.trim().length > 0 && !props.busy && !props.disabled)

function submit(): void {
  if (composing.value) return
  const value = text.value.trim()
  if (!canSend.value) return
  emit('feedback', { type: 'free_text', text: value })
  text.value = ''
}

function onEnter(event: KeyboardEvent): void {
  // Chinese IME composition: Enter confirms the composition, never sends.
  if (event.isComposing || composing.value) return
  event.preventDefault()
  submit()
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
        ✅ 确认这个国家
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
        @compositionstart="composing = true"
        @compositionend="composing = false"
        @keydown.enter="onEnter"
      >
      <button type="button" class="primary-btn send-btn" :disabled="!canSend" @click="submit">
        {{ busy ? '处理中…' : '发送' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { FeedbackPayload, HypothesisAction } from '@/types'
import { dimensionLabel } from '@/utils/clues'
import { answerPayload, groupActionOptions, type DialogueOption } from '@/utils/dialogue'

const props = defineProps<{
  action: HypothesisAction | null
  busy: boolean
  disabled?: boolean
  /**
   * Controlled draft. When provided the parent owns the text and clears it on
   * submit acknowledgment / restores it after a failure; omitted keeps local
   * state for standalone use.
   */
  draft?: string
}>()

const emit = defineEmits<{
  feedback: [payload: FeedbackPayload]
  'update:draft': [value: string]
}>()

const inputRef = ref<HTMLTextAreaElement | null>(null)
const localText = ref('')
const composing = ref(false)

const text = computed(() => (props.draft !== undefined ? props.draft : localText.value))
const optionGroups = computed(() => groupActionOptions(props.action))
const canSend = computed(() => text.value.trim().length > 0 && !props.busy && !props.disabled)

async function focusInput(): Promise<void> {
  await nextTick()
  inputRef.value?.focus()
}

function setText(value: string): void {
  if (props.draft !== undefined) emit('update:draft', value)
  else localText.value = value
}

function onInput(event: Event): void {
  setText((event.target as HTMLTextAreaElement).value)
}

// Auto-grow the textarea up to its CSS max-height so long scene descriptions stay readable.
watch(text, async () => {
  await nextTick()
  const element = inputRef.value
  if (!element) return
  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, 96)}px`
})

function submit(): void {
  if (composing.value) return
  const value = text.value.trim()
  if (!canSend.value) return
  // Never clear here: the parent acknowledges the submit into its optimistic
  // pipeline and restores the draft if the request fails.
  emit('feedback', { type: 'free_text', text: value })
}

function onEnter(event: KeyboardEvent): void {
  // Chinese IME composition: Enter confirms the composition, never sends.
  if (event.isComposing || composing.value) return
  // Shift+Enter inserts a newline; plain Enter sends.
  if (event.shiftKey) return
  event.preventDefault()
  submit()
}

function chooseOption(option: DialogueOption): void {
  if (props.busy || props.disabled) return
  emit('feedback', answerPayload(option))
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
        ✍️ 我来补充{{ action.payload?.dimensions?.length ? `（${action.payload.dimensions.map(dimensionLabel).join('、')}）` : '' }}
      </button>
      <span v-if="action.kind === 'done'" class="action-note">🎉 推理已完成</span>
    </div>

    <div v-if="optionGroups.length" class="answer-options">
      <div v-for="group in optionGroups" :key="group.dimension" class="answer-group">
        <span class="answer-dim">{{ group.label }}</span>
        <div class="answer-chips">
          <button
            v-for="option in group.options"
            :key="`${option.dimension}-${option.value}`"
            type="button"
            class="option-chip"
            :disabled="busy || disabled"
            @click="chooseOption(option)"
          >
            {{ option.label }}
          </button>
        </div>
      </div>
    </div>

    <div class="chat-input-row">
      <textarea
        ref="inputRef"
        :value="text"
        class="chat-input"
        rows="1"
        :placeholder="action?.kind === 'ask' || action?.kind === 'request_clue' ? '输入补充线索或直接描述场景…' : '输入反馈，或直接描述线索…'"
        :disabled="disabled || busy"
        @input="onInput"
        @compositionstart="composing = true"
        @compositionend="composing = false"
        @keydown.enter="onEnter"
      ></textarea>
      <button type="button" class="primary-btn send-btn" :disabled="!canSend" @click="submit">
        {{ busy ? '处理中…' : '发送' }}
      </button>
    </div>
  </div>
</template>
<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'
import type { ChatMessage, PendingChatMessage } from '@/types'
import { receiptFromMessage } from '@/utils/feedback'

const props = defineProps<{
  messages: ChatMessage[]
  /** Optimistic user turns that have not been reconciled yet. */
  pending?: PendingChatMessage[]
  loading?: boolean
}>()

const emit = defineEmits<{
  retry: [message: PendingChatMessage]
  dismiss: [message: PendingChatMessage]
}>()

const logRef = ref<HTMLElement | null>(null)

function scrollToBottom(): void {
  const element = logRef.value
  if (!element) return
  element.scrollTop = element.scrollHeight
}

function pendingKey(list: PendingChatMessage[] | undefined): string {
  return (list ?? [])
    .map((item) => `${item.id}:${item.status}:${item.error ?? ''}:${item.orphaned ? 1 : 0}`)
    .join('|')
}

/** Never repeat a receipt summary that the AI reply already contains. */
function summaryFor(message: ChatMessage): string | null {
  const summary = receiptFromMessage(message)?.summary
  if (!summary) return null
  const content = typeof message.content === 'string' ? message.content : ''
  return content.includes(summary) ? null : summary
}

// Keep the newest prompt visible: new messages, replaced content, optimistic
// bubbles and the typing indicator all pin the log to the bottom.
watch(
  () => [
    props.messages.length,
    props.messages.at(-1)?.content ?? '',
    props.loading,
    pendingKey(props.pending),
  ] as const,
  async () => {
    await nextTick()
    scrollToBottom()
  },
)

onMounted(scrollToBottom)
</script>

<template>
  <div ref="logRef" class="chat-log" aria-live="polite">
    <div class="msg ai">
      <div class="bubble">上传一张照片，开始定位。</div>
    </div>
    <template v-for="(message, index) in messages" :key="message.id ?? index">
      <div class="msg" :class="message.role">
        <div class="bubble">{{ message.content }}</div>
        <div
          v-if="receiptFromMessage(message)"
          class="receipt-chip"
          :class="receiptFromMessage(message)?.tone"
        >
          <span class="receipt-status">{{ receiptFromMessage(message)?.label }}</span>
          <span v-if="summaryFor(message)" class="receipt-summary">
            {{ summaryFor(message) }}
          </span>
          <span
            v-for="(detail, detailIndex) in receiptFromMessage(message)?.details"
            :key="detailIndex"
            class="receipt-detail"
          >
            {{ detail }}
          </span>
        </div>
      </div>
    </template>
    <div v-for="item in pending ?? []" :key="`pending-${item.id}`" class="msg user">
      <div class="bubble pending" :class="item.status">
        <span class="pending-text">{{ item.content }}</span>
        <span v-if="item.status === 'sending'" class="pending-state">发送中…</span>
        <template v-else>
          <span class="pending-state error">
            {{ item.orphaned ? '照片已更换' : '发送失败' }}{{ item.error ? `：${item.error}` : '' }}
          </span>
          <span class="pending-actions">
            <button
              type="button"
              class="retry-btn"
              :disabled="item.orphaned"
              :title="item.orphaned ? '照片已更换，请重新输入或删除' : '重新发送'"
              @click="emit('retry', item)"
            >
              重试
            </button>
            <button
              type="button"
              class="dismiss-btn"
              aria-label="删除这条未发送的消息"
              @click="emit('dismiss', item)"
            >
              ×
            </button>
          </span>
        </template>
      </div>
    </div>
    <div v-if="loading" class="msg ai">
      <div class="bubble typing">正在分析照片…</div>
    </div>
  </div>
</template>
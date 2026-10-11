<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'
import type { ChatMessage } from '@/types'

const props = defineProps<{
  messages: ChatMessage[]
  loading?: boolean
}>()

const logRef = ref<HTMLElement | null>(null)

function scrollToBottom(): void {
  const element = logRef.value
  if (!element) return
  element.scrollTop = element.scrollHeight
}

// Keep the newest prompt visible: new messages, replaced content and the
// typing indicator all pin the log to the bottom.
watch(
  () => [props.messages.length, props.messages.at(-1)?.content ?? '', props.loading] as const,
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
      <div class="bubble">你好！请上传一张街景或风景图，我会提取线索并与你一起推理拍摄地。</div>
    </div>
    <div v-for="(message, index) in messages" :key="message.id ?? index" class="msg" :class="message.role">
      <div class="bubble">{{ message.content }}</div>
    </div>
    <div v-if="loading" class="msg ai">
      <div class="bubble typing">AI 正在分析…</div>
    </div>
  </div>
</template>
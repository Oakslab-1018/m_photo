<script setup lang="ts">
import type { ChatMessage } from '@/types'

defineProps<{
  messages: ChatMessage[]
  loading?: boolean
}>()
</script>

<template>
  <div class="chat-log" aria-live="polite">
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

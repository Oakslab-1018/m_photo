<script setup lang="ts">
import type { Ref } from 'vue'
import { AnimatePresence, motion, useReducedMotion } from 'motion-v'
import ChatLog from '@/components/ChatLog.vue'
import type { ChatMessage, PendingChatMessage } from '@/types'

const props = defineProps<{
  open: boolean
  messages: ChatMessage[]
  pendingMessages: PendingChatMessage[]
  loading: boolean
  dialogRef: Ref<HTMLElement | null>
}>()

const emit = defineEmits<{
  close: []
  retry: [pending: PendingChatMessage]
  dismiss: [pending: PendingChatMessage]
}>()

const reduceMotion = useReducedMotion()
</script>

<template>
  <AnimatePresence>
    <motion.div
      v-if="open"
      key="history-layer"
      class="history-layer"
      :initial="reduceMotion ? false : { opacity: 0 }"
      :animate="{ opacity: 1 }"
      :exit="{ opacity: 0 }"
      :transition="{ duration: reduceMotion ? 0 : 0.18 }"
    >
      <div class="drawer-backdrop" @click="emit('close')" />
      <motion.div
        class="history-drawer-shell"
        :initial="reduceMotion ? false : { opacity: 0, x: 18 }"
        :animate="{ opacity: 1, x: 0 }"
        :transition="{ duration: reduceMotion ? 0 : 0.24, ease: [0.2, 0.7, 0.2, 1] }"
      >
        <aside
          id="reasoning-history-drawer"
          :ref="props.dialogRef"
          class="history-drawer"
          role="dialog"
          aria-modal="true"
          tabindex="-1"
          aria-labelledby="history-drawer-title"
        >
          <header class="history-drawer__head">
            <div>
              <span class="section-kicker">HISTORY</span>
              <h2 id="history-drawer-title">对话记录</h2>
              <p>对话与待发送消息</p>
            </div>
            <div class="history-drawer__actions">
              <span>{{ messages.length }} 条</span>
              <button type="button" class="icon-btn" data-modal-autofocus aria-label="关闭对话记录" @click="emit('close')">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                  <path d="M6 6l12 12M18 6 6 18" />
                </svg>
              </button>
            </div>
          </header>
          <div class="history-drawer__body">
            <ChatLog
              :messages="messages"
              :pending="pendingMessages"
              :loading="loading"
              tabindex="0"
              role="region"
              aria-label="完整对话记录，可滚动查看"
              @retry="emit('retry', $event)"
              @dismiss="emit('dismiss', $event)"
            />
          </div>
        </aside>
      </motion.div>
    </motion.div>
  </AnimatePresence>
</template>

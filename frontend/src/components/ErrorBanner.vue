<script setup lang="ts">
import { AnimatePresence, motion, useReducedMotion } from 'motion-v'

defineProps<{
  message: string | null
  inert: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

const reduceMotion = useReducedMotion()
</script>

<template>
  <AnimatePresence>
    <motion.div
      v-if="message"
      key="error-banner"
      class="error-banner"
      role="alert"
      :inert="inert"
      :initial="reduceMotion ? false : { opacity: 0, y: 10 }"
      :animate="{ opacity: 1, y: 0 }"
      :exit="{ opacity: 0, y: 6 }"
      :transition="{ duration: reduceMotion ? 0 : 0.2 }"
    >
      <span>{{ message }}</span>
      <button type="button" aria-label="关闭错误提示" @click="emit('close')">×</button>
    </motion.div>
  </AnimatePresence>
</template>

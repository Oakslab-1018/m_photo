<script setup lang="ts">
import { computed, type Ref } from 'vue'
import { motion, useReducedMotion } from 'motion-v'
import CandidatePanel from '@/components/CandidatePanel.vue'
import ChatLog from '@/components/ChatLog.vue'
import FeedbackBar from '@/components/FeedbackBar.vue'
import GuidedDialoguePanel from '@/components/GuidedDialoguePanel.vue'
import type {
  ChatMessage,
  CountryScore,
  FeedbackPayload,
  GuidedLocationLevel,
  HypothesisAction,
  LocationTarget,
  PendingChatMessage,
} from '@/types'
import type {
  CandidatePanelBindings,
  GuidedDialogueBindings,
  LiveStatus,
  MobileSheetState,
} from '@/components/uiTypes'

const props = defineProps<{
  visible: boolean
  collapsed: boolean
  wide: boolean
  isMobile: boolean
  sheetState: MobileSheetState
  inert: boolean
  guidedActive: boolean
  guidedLoading: boolean
  messages: ChatMessage[]
  pendingMessages: PendingChatMessage[]
  liveStatus: LiveStatus
  questionPending: boolean
  inferenceNotice: string | null
  guided: GuidedDialogueBindings
  candidates: CandidatePanelBindings
  feedbackAction: HypothesisAction | null
  feedbackDraft: string
  feedbackBusy: boolean
  hasSession: boolean
  legacyLoading: boolean
  rootRef: Ref<HTMLElement | null>
}>()

const emit = defineEmits<{
  expand: []
  collapse: []
  'toggle-width': []
  'sheet-state': [state: MobileSheetState]
  'update:feedback-draft': [value: string]
  feedback: [payload: FeedbackPayload]
  'retry-pending': [pending: PendingChatMessage]
  'dismiss-pending': [pending: PendingChatMessage]
  confirm: [score: CountryScore]
  reject: [score: CountryScore]
  'select-country': [iso: string]
  'select-guided-target': [target: LocationTarget]
  'confirm-guided-target': [target: LocationTarget]
  'reject-guided-target': [target: LocationTarget]
  'request-cities': [iso: string, sourceIso: string | null]
  'select-location': [target: LocationTarget]
  'confirm-location': [target: LocationTarget]
  'reject-location': [target: LocationTarget]
  'confirm-clue': [clueId: string]
  'reject-clue': [clueId: string]
  answer: [payload: { value: string; label: string }]
  unknown: []
  'retract-target': [target: LocationTarget]
  'retract-clue': [clueId: string]
  backtrack: [level: GuidedLocationLevel]
  'next-phase': []
  'back-to-review': []
  'review-back': []
  'free-text': [text: string]
  retry: []
}>()

const reduceMotion = useReducedMotion()
const enterFrom = computed(() => (reduceMotion.value ? false : { opacity: 0, y: 8 }))
const enterSettle = { opacity: 1, y: 0 }
const enterEase = { duration: 0.24, ease: [0.2, 0.7, 0.2, 1] as const }

</script>

<template>
  <aside
    v-show="visible"
    :ref="props.rootRef"
    class="dock-anchor reasoning-dock-anchor"
    :class="{ 'is-wide': wide, 'mobile-task-sheet': isMobile, 'is-collapsed': collapsed, 'is-guided-loading': guidedActive && guidedLoading }"
    :data-sheet-state="sheetState"
    :inert="inert"
    aria-label="定位任务"
  >
    <motion.button
      v-if="collapsed"
      type="button"
      class="reasoning-task-chip"
      aria-label="展开定位任务"
      :initial="enterFrom"
      :animate="enterSettle"
      :transition="enterEase"
      @click="emit('expand')"
    >
      <span class="reasoning-task-chip__icon" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">
          <path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z" />
          <path d="M8 10h8M8 14h5" />
        </svg>
      </span>
      <span class="reasoning-task-chip__copy">
        <span>定位任务</span>
        <strong>定位助手</strong>
        <small>{{ liveStatus.text }}</small>
      </span>
      <span class="live-dot" :class="liveStatus.tone" aria-hidden="true" />
    </motion.button>

    <motion.section
      v-show="!collapsed"
      class="floating-dock reasoning-dock"
      aria-labelledby="reasoning-dock-title"
      :initial="enterFrom"
      :animate="enterSettle"
      :transition="enterEase"
    >
      <header class="dock-head reasoning-dock__head">
        <div class="dock-heading">
          <span class="section-kicker">REASONING</span>
          <h2 id="reasoning-dock-title">定位助手</h2>
          <span class="dock-muted">{{ messages.length }} 条</span>
        </div>
        <div class="dock-head__actions">
          <div v-if="isMobile" class="sheet-state-switch" role="group" aria-label="任务面板高度">
            <button type="button" :class="{ active: sheetState === 'peek' }" :aria-pressed="sheetState === 'peek'" title="精简高度，保留输入框" @click="emit('sheet-state', 'peek')">精简</button>
            <button type="button" :class="{ active: sheetState === 'half' }" :aria-pressed="sheetState === 'half'" title="半屏展开任务" @click="emit('sheet-state', 'half')">半屏</button>
            <button type="button" :class="{ active: sheetState === 'full' }" :aria-pressed="sheetState === 'full'" title="全屏展开任务" @click="emit('sheet-state', 'full')">全屏</button>
          </div>
          <template v-else>
            <button
              type="button"
              class="icon-btn width-toggle"
              :class="{ active: wide }"
              :aria-pressed="wide"
              :aria-label="wide ? '还原面板宽度' : '加宽面板'"
              :title="wide ? '还原宽度' : '加宽到约 560px'"
              @click="emit('toggle-width')"
            >
              <svg v-if="!wide" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <path d="M4 5v14M20 5v14M8 12h8M11 9l3 3-3 3" />
              </svg>
              <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <path d="M4 5v14M20 5v14M8 12h8M13 9l-3 3 3 3" />
              </svg>
            </button>
            <button
              type="button"
              class="icon-btn"
              aria-label="收起助手面板"
              title="收起为任务卡"
              @click="emit('collapse')"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <path d="m9 6 6 6-6 6" />
              </svg>
            </button>
          </template>
        </div>
      </header>

      <motion.div
        :key="`${liveStatus.tone}:${liveStatus.text}`"
        class="reasoning-live"
        :class="liveStatus.tone"
        role="status"
        aria-live="polite"
        :initial="reduceMotion ? false : { opacity: 0.4 }"
        :animate="{ opacity: 1 }"
        :transition="{ duration: reduceMotion ? 0 : 0.2 }"
      >
        <span class="live-dot" aria-hidden="true" />
        <span>{{ liveStatus.text }}</span>
        <span v-if="questionPending" class="chat-alert">待回答</span>
      </motion.div>

      <div class="reasoning-body" :class="{ 'has-legacy-candidates': !guidedActive }">
        <motion.div
          v-if="inferenceNotice"
          class="inference-notice reasoning-notice"
          role="status"
          :initial="enterFrom"
          :animate="enterSettle"
          :transition="enterEase"
        >
          {{ inferenceNotice }}
        </motion.div>

        <div class="reasoning-grid">
          <div class="reasoning-primary">
            <GuidedDialoguePanel
              v-if="guidedActive"
              v-bind="guided"
              @select-location="emit('select-location', $event)"
              @confirm-location="emit('confirm-location', $event)"
              @reject-location="emit('reject-location', $event)"
              @confirm-clue="emit('confirm-clue', $event)"
              @reject-clue="emit('reject-clue', $event)"
              @answer="emit('answer', $event)"
              @unknown="emit('unknown')"
              @retract-target="emit('retract-target', $event)"
              @retract-clue="emit('retract-clue', $event)"
              @backtrack="emit('backtrack', $event)"
              @next-phase="emit('next-phase')"
              @back-to-review="emit('back-to-review')"
              @review-back="emit('review-back')"
              @free-text="emit('free-text', $event)"
              @retry="emit('retry')"
            />

            <div v-else class="legacy-reasoning">
              <div class="legacy-thread">
                <ChatLog
                  :messages="messages"
                  :pending="pendingMessages"
                  :loading="legacyLoading"
                  @retry="emit('retry-pending', $event)"
                  @dismiss="emit('dismiss-pending', $event)"
                />
              </div>
              <FeedbackBar
                :draft="feedbackDraft"
                :action="feedbackAction"
                :busy="feedbackBusy"
                :disabled="!hasSession"
                @update:draft="emit('update:feedback-draft', $event)"
                @feedback="emit('feedback', $event)"
              />
            </div>
          </div>

          <CandidatePanel
            v-if="!guidedActive"
            v-bind="candidates"
            surface="candidates"
            class="legacy-candidate-panel"
            :guided-mode="false"
            :guided-country-allowed="true"
            @confirm="emit('confirm', $event)"
            @reject="emit('reject', $event)"
            @select-country="emit('select-country', $event)"
            @select-guided-target="emit('select-guided-target', $event)"
            @confirm-guided-target="emit('confirm-guided-target', $event)"
            @reject-guided-target="emit('reject-guided-target', $event)"
            @request-cities="(iso, sourceIso) => emit('request-cities', iso, sourceIso)"
          />
        </div>
      </div>
    </motion.section>
  </aside>
</template>


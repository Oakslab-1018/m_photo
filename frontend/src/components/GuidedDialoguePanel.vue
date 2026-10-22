<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { AnimatePresence, motion, useReducedMotion } from 'motion-v'
import GuidedCityClueImpact from '@/components/GuidedCityClueImpact.vue'
import PlaceMatchEvidence from '@/components/PlaceMatchEvidence.vue'
import type {
  ConfidenceRoute,
  GuidedDialogue,
  GuidedEventReceipt,
  GuidedLocationLevel,
  GuidedPhase,
  HypothesisAction,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import {
  useGuidedComposer,
  type GuidedFreeTextSubmission,
} from '@/composables/useGuidedComposer'
import {
  GUIDED_STEPS,
  candidatePointLabel,
  candidateScoreLabel,
  guidedCandidateComposition,
  guidedCityClueImpact,
  guidedCurrentReviewClue,
  guidedPreviousLevel,
  guidedQuestionMessage,
  guidedQuestionReason,
  guidedQuestionTitle,
  guidedReviewProgress,
  guidedReviewReady,
  guidedStageTitle,
  guidedStepState,
  guidedTargetKey,
  isBlockingGuidedQuestion,
  locationLevelLabel,
  reviewedClueText,
  safeGuidedReceipt,
  safeGuidedText,
  selectTopGuidedCandidates,
  targetLabel,
  type CountryConfidenceShortcut,
} from '@/utils/guided'

const props = withDefaults(
  defineProps<{
    dialogue: GuidedDialogue | null
    candidates: LocationCandidate[]
    /** Country-only confidence route from the same authoritative snapshot. */
    confidence?: ConfidenceRoute | null
    /** Action kind must match the confidence action; the panel never re-routes tiers. */
    action?: HypothesisAction | null
    /** Precomputed country shortcut from the composable. */
    confidenceShortcut?: CountryConfidenceShortcut | null
    /** Server-backed targets selected by the confidence top3 order. */
    confidenceCandidates?: LocationCandidate[]
    loading?: boolean
    applying?: boolean
    /** Legacy point/feedback/upload work that also mutates server evidence. */
    externalBusy?: boolean
    error?: string | null
    conflict?: string | null
    canRetry?: boolean
    receipt?: GuidedEventReceipt | null
    selectedGuidedTargetKey?: string | null
    suggestedFreeText?: string | null
    /** Parent acknowledgement for the last guided free-text submission. */
    freeTextSubmission?: GuidedFreeTextSubmission | null
    /**
     * UI phase: `review` renders exactly one server-pinned observation/question
     * and never any candidate; `candidates` unlocks the explicit Top-5 step.
     */
    phase?: GuidedPhase
    /** Server rankings unlocked (sticky across a completed review round). */
    candidatesUnlocked?: boolean
    /** True when the review history can restore the previous observation. */
    canReviewBack?: boolean
    /** New observations may temporarily block only the location actions. */
    locationConfirmationReady?: boolean
    /** Newly appended clue ids are highlighted in the single review box. */
    recentClueIds?: string[]
  }>(),
  {
    confidence: null,
    action: null,
    confidenceShortcut: null,
    confidenceCandidates: () => [],
    loading: false,
    applying: false,
    externalBusy: false,
    error: null,
    conflict: null,
    canRetry: false,
    receipt: null,
    selectedGuidedTargetKey: null,
    suggestedFreeText: null,
    freeTextSubmission: null,
    phase: 'review',
    candidatesUnlocked: true,
    canReviewBack: false,
    locationConfirmationReady: true,
    recentClueIds: () => [],
  },
)

const emit = defineEmits<{
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
  'free-text': [text: string]
  retry: []
  'next-phase': []
  'back-to-review': []
  'review-back': []
}>()

const backtrackPrompt = ref<GuidedLocationLevel | null>(null)
const panelRef = ref<HTMLElement | null>(null)
const reduceMotion = useReducedMotion()
/** Initial state shared by dock content entering after a state change. */
const enterState = computed(() => (reduceMotion.value ? false : { opacity: 0, y: 6 }))
const enterTarget = { opacity: 1, y: 0 }
const enterTransition = { duration: 0.22, ease: [0.2, 0.7, 0.2, 1] as const }
const searchInputRef = ref<HTMLInputElement | null>(null)
const questionSectionRef = ref<HTMLElement | null>(null)
const candidateSectionRef = ref<HTMLElement | null>(null)

const stage = computed(() => props.dialogue?.stage ?? 'country')
const busy = computed(() => props.loading || props.applying || props.externalBusy)
const {
  text: composerText,
  composing,
  canSend: canSendComposer,
  submit: submitComposer,
  onEnter: onComposerEnter,
} = useGuidedComposer({
  suggestedText: () => props.suggestedFreeText,
  submission: () => props.freeTextSubmission,
  isBusy: () => busy.value,
  onSend: (text) => emit('free-text', text),
})
const isResolved = computed(() => props.dialogue?.stage === 'resolved')
const stageTitle = computed(() => guidedStageTitle(stage.value))
const pendingQuestion = computed(() => props.dialogue?.pending_question ?? null)
const pendingClueCount = computed(
  () => props.dialogue?.clues.filter((clue) => clue.status === 'pending').length ?? 0,
)

/** One server-pinned observation; the UI never iterates local clue order. */
const currentReviewClue = computed(() => guidedCurrentReviewClue(props.dialogue))
const reviewProgress = computed(() => guidedReviewProgress(props.dialogue))
const reviewReady = computed(() => guidedReviewReady(props.dialogue))
const isReviewClueQuestion = computed(() => pendingQuestion.value?.kind === 'review_clue')
const blockingQuestion = computed(() =>
  isBlockingGuidedQuestion(pendingQuestion.value) ? pendingQuestion.value : null,
)
/** Candidate surfaces render only in the explicit 下一步 phase. */
const candidatesPhase = computed(
  () => props.phase === 'candidates' && props.candidatesUnlocked && !isResolved.value,
)
/** Immediate country route; lower stages never inherit country confidence. */
const countryShortcut = computed(() =>
  stage.value === 'country' ? props.confidenceShortcut ?? null : null,
)
const shortcutCandidates = computed(() => {
  const mode = countryShortcut.value
  if (!mode || stage.value !== 'country') return []
  return props.confidenceCandidates.slice(0, mode === 'high' ? 1 : 3)
})
const shortcutExplanation = computed(() => {
  const message = safeGuidedText(props.action?.message)
  if (message) return message
  return countryShortcut.value === 'high'
    ? '最符合这张照片的国家。'
    : '几个候选比较接近，请按照片选择。'
})
const shortcutTitle = computed(() => {
  if (countryShortcut.value === 'high') return '推荐国家 / 地区'
  if (props.confidence?.tier === 'low') return '候选仅供参考'
  return '国家 / 地区候选待确认'
})
const shortcutNote = computed(() => {
  if (countryShortcut.value === 'high') return '确认后可继续选城市'
  if (props.confidence?.tier === 'low') return '只在确定时选择'
  return '请选择最符合照片的'
})
const shortcutFooter = computed(() => {
  if (countryShortcut.value === 'high') {
    return '确认国家后，继续选择城市。'
  }
  if (props.confidence?.tier === 'low') {
    return '这些只是参考；不确定时请继续补充线索。'
  }
  return '请从下面选择并确认国家。'
})
const showLowCluePrompt = computed(() => {
  const kind = pendingQuestion.value?.kind
  return (
    stage.value === 'country' &&
    props.confidence?.tier === 'low' &&
    !countryShortcut.value &&
    (kind === 'ask' || kind === 'request_clue')
  )
})
/** Low-country narration is shown only for the exact authoritative question. */
const narratedLowQuestion = computed(() => {
  if (
    countryShortcut.value ||
    stage.value !== 'country' ||
    props.confidence?.tier !== 'low'
  ) {
    return ''
  }
  const question = pendingQuestion.value
  const action = props.action
  if (!question || !action) return ''
  if (question.kind !== 'ask' && question.kind !== 'request_clue') return ''
  if (action.kind !== question.kind) return ''
  const payload = action.payload
  if (payload?.question_id !== question.id || payload?.question_kind !== question.kind) return ''
  const message = safeGuidedText(action.message).trim()
  return message && message !== questionMessage.value ? message : ''
})
/** Prominent backend backtrack target for city/resolved. */
const previousLevel = computed(() => guidedPreviousLevel(props.dialogue?.stage))
const nextStepLabel = computed(() =>
  stage.value === 'resolved' ? '' : `下一步：查看${locationLevelLabel(stage.value)}候选`,
)

const currentTaskLabel = computed(() => {
  if (isResolved.value) return '定位完成'
  if (countryShortcut.value) {
    return props.confidence?.tier === 'low' ? '选择候选' : '确认国家'
  }
  if (props.phase === 'candidates') return stage.value === 'city' ? '选择城市' : '选择国家 / 地区'
  if (isReviewClueQuestion.value) {
    const progress = reviewProgress.value
    return progress ? `核对观察 ${progress.current}/${progress.total}` : '核对观察'
  }
  if (blockingQuestion.value) return guidedQuestionTitle(blockingQuestion.value)
  if (pendingClueCount.value > 0) return `待核对 ${pendingClueCount.value} 条`
  return reviewReady.value ? '查看候选' : '核对观察'
})

const progressText = computed(() => {
  if (isResolved.value) return '2 / 2 · 已完成'
  const current = GUIDED_STEPS.findIndex((step) => step.level === stage.value)
  return `${Math.max(1, current + 1)} / ${GUIDED_STEPS.length} · 当前步骤`
})
const compactProgressText = computed(() => progressText.value.split(' · ')[0])

const pathSteps = computed(() =>
  GUIDED_STEPS.map((step) => ({
    ...step,
    target: props.dialogue?.confirmed[step.level] ?? null,
    state: props.dialogue ? guidedStepState(props.dialogue.stage, step.level) : 'pending',
  })),
)

const questionOptions = computed(() => {
  const question = pendingQuestion.value
  if (!question) return []
  // confirm_location options are catalog ids resolved through the candidate
  // cards; they are never rendered as answer chips.
  if (question.kind === 'confirm_location') return []
  return (question.options ?? []).filter(
    (option) => typeof option?.value === 'string' && typeof option?.label === 'string',
  )
})

const answerChoiceOptions = computed(() => {
  const kind = pendingQuestion.value?.kind
  if (kind !== 'ask' && kind !== 'request_clue') return []
  return questionOptions.value
})

/** Defense in depth: even a caller passing the full server list cannot render past Top-5. */
const topCandidates = computed(() =>
  !candidatesPhase.value
    ? []
    : selectTopGuidedCandidates(props.candidates, stage.value),
)

function isRecentClue(clueId: string): boolean {
  return props.recentClueIds.includes(clueId)
}

const candidateTitle = computed(() => {
  if (stage.value === 'country') return '国家 / 地区候选'
  if (stage.value === 'city') return '城市候选'
  return '候选地点'
})

const candidateComposition = computed(() => guidedCandidateComposition(topCandidates.value))
const candidateLevelName = computed(() => (stage.value === 'city' ? '城市' : '国家 / 地区'))
const hasCityImpactDetails = computed(
  () =>
    stage.value === 'city' &&
    topCandidates.value.some((candidate) => guidedCityClueImpact(candidate)?.hasDetail),
)
const candidateSectionNote = computed(() => {
  const composition = candidateComposition.value
  // City scores already share one ordering. User text/catalog provenance must
  // not be described as priority over model scores.
  if (stage.value === 'city') {
    if (composition.hasModelScores) return '按匹配度排序'
    if (composition.hasExplicitSelectionBasis) return '按目录顺序'
    return `目录候选（${topCandidates.value.length} 项）`
  }
  if (composition.hasUserText) {
    return composition.mixed ? '含线索匹配' : '线索匹配候选'
  }
  if (composition.hasCatalogSearch) {
    return composition.mixed ? '含目录检索' : '目录检索候选'
  }
  if (composition.hasModelScores) return '候选前五'
  return `目录候选（${topCandidates.value.length} 项）`
})
const candidateSectionHint = computed(() => {
  const composition = candidateComposition.value
  if (!props.locationConfirmationReady) {
    return '有新的观察待核对，暂时不能确认或排除。'
  }
  if (stage.value === 'city') {
    if (composition.hasModelScores) {
      return hasCityImpactDetails.value
        ? '可展开查看线索如何影响排序。'
        : '匹配度不是概率。'
    }
    return '暂无评分，按目录顺序显示；可补充线索或输入城市名。'
  }
  if (composition.hasUserText) {
    return '标有“线索匹配候选”的条目来自你输入的文字。'
  }
  if (composition.hasCatalogSearch) {
    return '目录检索命中的条目优先显示。'
  }
  if (composition.hasModelScores) {
    return '匹配度仅供参考。'
  }
  return `按目录顺序显示；可输入${candidateLevelName.value}名称，或补充照片里的线索。`
})

const displayError = computed(() => props.conflict ?? props.error)
const questionMessage = computed(() => {
  const question = pendingQuestion.value
  if (!question) return ''
  return guidedQuestionMessage(question, currentReviewClue.value, candidatesPhase.value)
})
const questionReason = computed(() => {
  const question = pendingQuestion.value
  return question ? guidedQuestionReason(question) : ''
})
const latestAssistantMessage = computed(() => {
  if (isResolved.value) {
    return safeGuidedText(props.action?.message).trim()
      || '定位已完成，还可以继续补充线索。'
  }
  if (countryShortcut.value) return shortcutExplanation.value
  // In the low-confidence country route, only the validated question may be
  // repeated near the composer. Never surface a stale action that could name
  // candidates from an earlier hidden ranking.
  if (stage.value === 'country' && props.confidence?.tier === 'low') {
    return narratedLowQuestion.value || questionMessage.value
  }
  // A country confidence object is country-only. Below country it is treated
  // as stale and never reused as composer narration; otherwise the current
  // server action explanation is useful conversational context.
  if (props.confidence) return questionMessage.value
  return safeGuidedText(props.action?.message).trim() || questionMessage.value
})
const composerHint = computed(() => {
  if (isResolved.value) {
    return '还可以补充线索；新增线索会重新核对，不会改变已确认结果。'
  }
  if (!props.candidatesUnlocked) {
    return '输入照片里的细节，如河边、红砖建筑、招牌文字；不会自动确认地点。'
  }
  if (stage.value === 'country') {
    return '输入照片里的细节，或直接输入国家/地区、店名；不会自动确认地点。'
  }
  return '输入照片里的细节，或直接输入城市名、路牌；不会自动确认地点。'
})
const latestReceiptText = computed(() => safeGuidedReceipt(props.receipt?.summary))
const receiptStatusText = computed(() => {
  const summary = latestReceiptText.value
  if (summary) return summary
  const status = props.receipt?.status?.trim().toLowerCase()
  if (status === 'unchanged') return '这条线索之前已保存。'
  if (status === 'unrecognized') return '没有识别到新线索。'
  return '已更新'
})
const submissionReceiptText = computed(() =>
  safeGuidedReceipt(props.freeTextSubmission?.receipt?.summary),
)
const composerStatusText = computed(() => {
  const submission = props.freeTextSubmission
  if (!submission) return ''
  if (submission.status === 'pending') return '正在处理…'
  if (submission.status === 'failed') {
    const reason = safeGuidedText(submission.error).trim()
    return `${reason || '这次提交没有完成'} · 草稿已保留，可修改后再发送。`
  }
  if (submission.status === 'unconfirmed') {
    const reason = safeGuidedText(submission.error).trim()
    return `${reason || '没有收到确认'} · 草稿已保留，请核对后再试。`
  }
  if (submission.status === 'unchanged') {
    return '这条线索之前已记录。'
  }
  return submissionReceiptText.value
    ? `助手：${submissionReceiptText.value}`
    : '已收到。'
})

/** Scroll only the guided panel; never scroll the document or a candidate card. */
function scrollPanelTo(target: HTMLElement | null): void {
  const panel = panelRef.value
  if (!panel || !target) return
  const panelRect = panel.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const nextTop = panel.scrollTop + targetRect.top - panelRect.top - 8
  const maxTop = Math.max(0, panel.scrollHeight - panel.clientHeight)
  panel.scrollTop = Math.min(Math.max(0, nextTop), maxTop)
}

function scrollToQuestionOrCandidates(): void {
  void nextTick(() => {
    const target = questionSectionRef.value ?? candidateSectionRef.value
    if (target) {
      scrollPanelTo(target)
      return
    }
    if (panelRef.value) panelRef.value.scrollTop = 0
  })
}

function scrollToCandidates(): void {
  void nextTick(() => {
    scrollPanelTo(candidateSectionRef.value)
  })
}

function requestBacktrack(level: GuidedLocationLevel): void {
  backtrackPrompt.value = level
}

function confirmBacktrack(): void {
  const level = backtrackPrompt.value
  if (!level || busy.value) return
  backtrackPrompt.value = null
  emit('backtrack', level)
}

function requestRetry(): void {
  if (busy.value) return
  emit('retry')
}

function focusFreeText(): void {
  void nextTick(() => searchInputRef.value?.focus())
}

function selectCandidate(target: LocationTarget): void {
  if (busy.value) return
  emit('select-location', target)
}

function isCandidateSelected(target: LocationTarget): boolean {
  const key = guidedTargetKey(target)
  return key !== null && key === props.selectedGuidedTargetKey
}

function candidateTargetKey(target: LocationTarget): string {
  return guidedTargetKey(target) ?? target.id
}

watch(
  () => props.dialogue?.revision,
  () => {
    backtrackPrompt.value = null
  },
)

// Only an explicit phase flip (下一步 / 返回核对 / 显式返回) moves the panel
// viewport; a review round never auto-scrolls towards hidden candidates.
watch(
  () => props.phase,
  (next, previous) => {
    if (next === previous) return
    if (next === 'candidates') scrollToCandidates()
    else scrollToQuestionOrCandidates()
  },
  { flush: 'post' },
)

watch(stage, (nextStage, previousStage) => {
  if (!props.dialogue || !previousStage || nextStage === previousStage) return
  scrollToQuestionOrCandidates()
})
</script>

<template>
  <section class="guided-dialogue-shell">
    <div ref="panelRef" class="guided-panel">
      <header class="guided-dock-header">
        <details class="guided-dock-details">
          <summary
            class="guided-dock-summary"
            :aria-label="`${stageTitle}，${progressText}，查看路径和助手详情`"
          >
            <h2 id="guided-stage-heading">{{ stageTitle }}</h2>
            <span class="guided-progress" aria-hidden="true">{{ compactProgressText }}</span>
            <span class="guided-details-toggle" aria-hidden="true">详情</span>
          </summary>

          <div class="guided-dock-details-body">
            <div class="guided-details-heading">
              <span>{{ progressText }}</span>
              <span role="status" aria-live="polite">{{ currentTaskLabel }}</span>
            </div>

            <div v-if="previousLevel && dialogue" class="guided-details-backtrack">
              <button
                type="button"
                class="ghost-btn guided-back-btn"
                :disabled="busy"
                :title="`回到上一层级：${locationLevelLabel(previousLevel)}（保留其他结果和排除）`"
                @click="emit('backtrack', previousLevel)"
              >
                返回上一步：{{ locationLevelLabel(previousLevel) }}
              </button>
            </div>

            <ol class="guided-path" aria-label="已确认路径">
              <li
                v-for="(step, index) in pathSteps"
                :key="step.level"
                :class="`is-${step.state}`"
              >
                <span class="guided-step-index" aria-hidden="true">{{ index + 1 }}</span>
                <span class="guided-step-main">
                  <span class="guided-step-label">{{ step.label }}</span>
                  <strong>{{ step.target ? targetLabel(step.target) : '待确认' }}</strong>
                </span>
                <button
                  v-if="step.target && !busy"
                  type="button"
                  class="guided-backtrack-btn"
                  :title="`返回${step.label}会清除后续确认`"
                  :aria-label="`返回${step.label}，会清除后续确认`"
                  @click="requestBacktrack(step.level)"
                >
                  返回
                </button>
              </li>
            </ol>

            <div v-if="dialogue && latestAssistantMessage" class="guided-task-message">
              <span class="guided-task-message-label">最新助手提示</span>
              <p>{{ latestAssistantMessage }}</p>
            </div>

            <div v-if="backtrackPrompt" class="guided-backtrack-confirm" role="alert">
              <span>后续已确认层级会被清除，继续？</span>
              <div>
                <button type="button" class="ghost-btn" :disabled="busy" @click="backtrackPrompt = null">
                  取消
                </button>
                <button type="button" class="mini-btn reject" :disabled="busy" @click="confirmBacktrack">
                  确认返回
                </button>
              </div>
            </div>
          </div>
        </details>
      </header>

      <div class="guided-main">
        <div v-if="displayError" class="guided-error" role="alert">
          <p>{{ displayError }}</p>
          <button type="button" class="ghost-btn" :disabled="busy" @click="requestRetry">
            {{ canRetry ? '重试上次操作' : '重新加载' }}
          </button>
        </div>

        <div v-if="loading && !dialogue" class="guided-loading" role="status" aria-live="polite">
          正在加载…
        </div>

        <template v-else-if="dialogue">
          <motion.div
            v-if="receipt?.summary || receipt?.status"
            class="guided-receipt"
            role="status"
            :initial="enterState"
            :animate="enterTarget"
            :transition="enterTransition"
          >
            {{ receiptStatusText }}
          </motion.div>

          <motion.div
            v-if="isResolved"
            class="guided-resolved"
            :initial="enterState"
            :animate="enterTarget"
            :transition="enterTransition"
          >
            <span class="guided-resolved-label">最终结果</span>
            <strong>{{ targetLabel(dialogue.confirmed.city) }}</strong>
            <p>已按你的确认完成定位。城市位置是代表点，不是照片 GPS。</p>
          </motion.div>

          <motion.section
            v-if="!isResolved && countryShortcut"
            class="guided-confidence-result"
            :class="'is-' + countryShortcut"
            :initial="enterState"
            :animate="enterTarget"
            :transition="enterTransition"
          >
            <div class="guided-confidence-head">
              <div>
                <span class="guided-confidence-kicker">国家 / 地区预测</span>
                <strong>{{ shortcutTitle }}</strong>
              </div>
              <span class="guided-confidence-note">{{ shortcutNote }}</span>
            </div>
            <p class="guided-confidence-explanation">{{ shortcutExplanation }}</p>

            <motion.article
              v-for="(candidate, index) in shortcutCandidates"
              :key="candidateTargetKey(candidate.target)"
              class="guided-candidate-card guided-confidence-card"
              :class="{ selected: isCandidateSelected(candidate.target) }"
              :initial="enterState"
              :animate="enterTarget"
              :transition="{ duration: 0.2, delay: Math.min(index, 5) * 0.035 }"
            >
              <div class="guided-candidate-summary">
                <button
                  type="button"
                  class="guided-candidate-select"
                  :disabled="busy"
                  :aria-current="isCandidateSelected(candidate.target) ? 'true' : undefined"
                  :aria-label="`选择候选 ${index + 1}：${targetLabel(candidate.target)}，在地图查看`"
                  @click="selectCandidate(candidate.target)"
                >
                  <span class="guided-candidate-number" aria-hidden="true">{{ index + 1 }}</span>
                  <span class="guided-candidate-main">
                    <strong>{{ targetLabel(candidate.target) }}</strong>
                    <span class="guided-candidate-name-meta">
                      <span class="guided-candidate-meta" :title="candidateScoreLabel(candidate)">
                        {{ candidateScoreLabel(candidate) }}
                      </span>
                      <span v-if="candidatePointLabel(candidate.target)" class="guided-candidate-point">
                        {{ candidatePointLabel(candidate.target) }}
                      </span>
                      <span v-else class="guided-candidate-point missing">无地图坐标</span>
                    </span>
                  </span>
                </button>
              </div>

              <GuidedCityClueImpact :candidate="candidate" />

              <details
                v-if="candidate.reason || candidate.place_match || !candidatePointLabel(candidate.target)"
                class="guided-candidate-detail"
              >
                <summary>依据说明</summary>
                <div class="guided-candidate-detail-body">
                  <p v-if="candidate.reason" class="guided-candidate-reason">{{ candidate.reason }}</p>
                  <p v-if="!candidatePointLabel(candidate.target)" class="guided-candidate-point-note">
                    没有地图坐标，选中后只显示所在区域。
                  </p>
                  <PlaceMatchEvidence v-if="candidate.place_match" :match="candidate.place_match" />
                </div>
              </details>

              <div class="guided-candidate-actions">
                <button
                  type="button"
                  class="mini-btn confirm"
                  :disabled="busy || !locationConfirmationReady"
                  @click="emit('confirm-location', candidate.target)"
                >
                  确认{{ locationLevelLabel(candidate.target.level) }}
                </button>
                <button
                  type="button"
                  class="mini-btn reject"
                  :disabled="busy || !locationConfirmationReady"
                  @click="emit('reject-location', candidate.target)"
                >
                  排除{{ locationLevelLabel(candidate.target.level) }}
                </button>
              </div>
            </motion.article>

            <div v-if="!shortcutCandidates.length" class="guided-empty">
              候选还没准备好，请重新加载。
            </div>
            <div class="guided-confidence-actions">
              <button
                v-if="countryShortcut === 'high'"
                type="button"
                class="ghost-btn"
                :disabled="busy"
                @click="focusFreeText"
              >
                补充线索
              </button>
              <span>{{ shortcutFooter }}</span>
            </div>
          </motion.section>

          <section
            v-if="!isResolved && !countryShortcut && phase === 'review'"
            ref="questionSectionRef"
            class="guided-question"
          >
            <template v-if="phase === 'review'">
              <div v-if="showLowCluePrompt" class="guided-low-confidence-note">
                <strong>信息不足，先补充照片里的线索</strong>
                <span>回答下面的问题，或在下方输入线索。</span>
              </div>
              <div v-if="isReviewClueQuestion" class="guided-review-box">
                <div class="guided-section-head">
                  <span class="guided-section-title">核对观察</span>
                  <span v-if="reviewProgress" class="guided-review-progress">
                    第 {{ reviewProgress.current }} / {{ reviewProgress.total }}
                  </span>
                  <span v-else class="guided-one-question">本轮只有这一条</span>
                </div>
                <template v-if="currentReviewClue">
                  <p class="guided-question-message">{{ questionMessage }}</p>
                  <p class="guided-question-reason">{{ questionReason }}</p>
                  <div class="guided-question-clue">
                    <strong>{{ reviewedClueText(currentReviewClue) }}</strong>
                    <span>{{ currentReviewClue.description || '暂无说明' }}</span>
                    <span v-if="isRecentClue(currentReviewClue.id)" class="guided-clue-new">新线索</span>
                  </div>
                  <div class="guided-inline-actions">
                    <button
                      type="button"
                      class="mini-btn confirm"
                      :disabled="busy"
                      @click="emit('confirm-clue', currentReviewClue.id)"
                    >
                      准确
                    </button>
                    <button
                      type="button"
                      class="mini-btn reject"
                      :disabled="busy"
                      @click="emit('reject-clue', currentReviewClue.id)"
                    >
                      不准确
                    </button>
                  </div>
                  <button
                    type="button"
                    class="guided-unknown-btn"
                    :disabled="busy"
                    @click="emit('unknown')"
                  >
                    不确定（不代表排除）
                  </button>
                </template>
                <div v-else class="guided-question-hint">
                  这条观察暂时无法显示，请重新加载。
                </div>
              </div>

              <div v-else-if="blockingQuestion" class="guided-review-box">
                <div class="guided-section-head">
                  <span class="guided-section-title">{{ guidedQuestionTitle(blockingQuestion) }}</span>
                  <span class="guided-one-question">本轮只有这一题</span>
                </div>
                <p v-if="narratedLowQuestion" class="guided-question-reason">
                  {{ narratedLowQuestion }}
                </p>
                <p class="guided-question-message">{{ questionMessage }}</p>
                <p class="guided-question-reason">{{ guidedQuestionReason(blockingQuestion) }}</p>
                <div v-if="answerChoiceOptions.length" class="guided-answer-options">
                  <button
                    v-for="option in answerChoiceOptions"
                    :key="option.value"
                    type="button"
                    class="option-chip"
                    :disabled="busy"
                    @click="emit('answer', { value: option.value, label: option.label })"
                  >
                    {{ option.label }}
                  </button>
                </div>
                <div v-else class="guided-question-hint">
                  在下方输入你能看到的线索。
                </div>
                <button
                  type="button"
                  class="guided-unknown-btn"
                  :disabled="busy"
                  @click="emit('unknown')"
                >
                  不确定，跳过
                </button>
              </div>

              <div v-else-if="pendingClueCount > 0" class="guided-question-hint">
                还有 {{ pendingClueCount }} 条观察待核对。
              </div>

              <div v-else class="guided-next-step" role="status">
                <div>
                  <strong>核对完成</strong>
                  <span>查看候选，选中符合照片的地点再确认。</span>
                  <span v-if="questionMessage" class="guided-question-hint">{{ questionMessage }}</span>
                </div>
                <button
                  type="button"
                  class="primary-btn guided-next-step-action"
                  :disabled="busy || !reviewReady"
                  @click="emit('next-phase')"
                >
                  {{ nextStepLabel }}
                </button>
              </div>

              <div class="guided-review-nav">
                <button
                  type="button"
                  class="ghost-btn"
                  :disabled="busy || !canReviewBack"
                  :title="canReviewBack ? '重新核对上一条已确认的观察' : '还没有已确认的观察'"
                  @click="emit('review-back')"
                >
                  重新核对上一条
                </button>
              </div>
            </template>
          </section>

          <section
            v-if="candidatesPhase && !countryShortcut"
            ref="candidateSectionRef"
            class="guided-candidates"
          >
            <div class="guided-section-head">
              <span class="guided-section-title">{{ candidateTitle }}</span>
              <span class="guided-section-note">{{ candidateSectionNote }}</span>
            </div>

            <div class="guided-candidates-toolbar">
              <button type="button" class="ghost-btn" :disabled="busy" @click="emit('back-to-review')">
                返回核对
              </button>
              <button
                type="button"
                class="ghost-btn"
                :disabled="busy || !canReviewBack"
                :title="canReviewBack ? '重新核对上一条已确认的观察' : '还没有已确认的观察'"
                @click="emit('review-back')"
              >
                重新核对上一条
              </button>
            </div>

            <div v-if="pendingQuestion?.kind === 'confirm_location'" class="guided-question-hint">
              请选择并确认，不会自动采用第一项。
            </div>

            <p class="guided-section-hint">{{ candidateSectionHint }}</p>
            <div v-if="!topCandidates.length" class="guided-empty">
              暂无候选；可补充照片里的招牌、地名或文字。
            </div>

            <motion.article
              v-for="(candidate, index) in topCandidates"
              :key="candidateTargetKey(candidate.target)"
              class="guided-candidate-card"
              :class="{ selected: isCandidateSelected(candidate.target) }"
              :initial="enterState"
              :animate="enterTarget"
              :transition="{ duration: 0.2, delay: Math.min(index, 5) * 0.035 }"
            >
              <div class="guided-candidate-summary">
                <button
                  type="button"
                  class="guided-candidate-select"
                  :disabled="busy"
                  :aria-current="isCandidateSelected(candidate.target) ? 'true' : undefined"
                  :aria-label="`选择候选 ${index + 1}：${targetLabel(candidate.target)}，在地图查看`"
                  @click="selectCandidate(candidate.target)"
                >
                  <span class="guided-candidate-number" aria-hidden="true">{{ index + 1 }}</span>
                  <span class="guided-candidate-main">
                    <strong>{{ targetLabel(candidate.target) }}</strong>
                    <span class="guided-candidate-name-meta">
                      <span class="guided-candidate-meta" :title="candidateScoreLabel(candidate)">
                        {{ candidateScoreLabel(candidate) }}
                      </span>
                      <span v-if="candidatePointLabel(candidate.target)" class="guided-candidate-point">
                        {{ candidatePointLabel(candidate.target) }}
                      </span>
                      <span v-else class="guided-candidate-point missing">无地图坐标</span>
                    </span>
                  </span>
                </button>
              </div>

              <GuidedCityClueImpact :candidate="candidate" />

              <details
                v-if="candidate.reason || candidate.place_match || !candidatePointLabel(candidate.target)"
                class="guided-candidate-detail"
              >
                <summary>依据说明</summary>
                <div class="guided-candidate-detail-body">
                  <p v-if="candidate.reason" class="guided-candidate-reason">{{ candidate.reason }}</p>
                  <p v-if="!candidatePointLabel(candidate.target)" class="guided-candidate-point-note">
                    没有地图坐标，选中后只显示所在区域。
                  </p>
                  <PlaceMatchEvidence v-if="candidate.place_match" :match="candidate.place_match" />
                </div>
              </details>

              <div class="guided-candidate-actions">
                <button
                  type="button"
                  class="mini-btn confirm"
                  :disabled="busy || !locationConfirmationReady"
                  @click="emit('confirm-location', candidate.target)"
                >
                  确认{{ locationLevelLabel(candidate.target.level) }}
                </button>
                <button
                  type="button"
                  class="mini-btn reject"
                  :disabled="busy || !locationConfirmationReady"
                  @click="emit('reject-location', candidate.target)"
                >
                  排除{{ locationLevelLabel(candidate.target.level) }}
                </button>
              </div>
            </motion.article>
          </section>

          <section v-if="candidatesPhase && dialogue.exclusions.length" class="guided-exclusions">
            <div class="guided-section-head">
              <span class="guided-section-title">已排除地点</span>
              <span class="guided-section-note">可撤回</span>
            </div>
            <p class="guided-section-hint">只排除这一项，可随时撤回。</p>
            <div
              v-for="target in dialogue.exclusions"
              :key="`${target.level}-${target.id}`"
              class="guided-exclusion-row"
            >
              <span>{{ locationLevelLabel(target.level) }} · {{ targetLabel(target) }}</span>
              <button
                type="button"
                class="guided-link-btn"
                :disabled="busy"
                @click="emit('retract-target', target)"
              >
                撤回排除
              </button>
            </div>
          </section>
        </template>

        <div v-else-if="!loading" class="guided-loading">
          还没加载好，请重试。
        </div>
      </div>
    </div>

    <form
      v-if="dialogue"
      class="guided-composer"
      :class="{ 'is-busy': busy }"
      aria-labelledby="guided-composer-label"
      :aria-busy="busy ? 'true' : 'false'"
      @submit.prevent="submitComposer"
    >
      <label id="guided-composer-label" for="guided-conversation-input" class="sr-only">
        补充线索
      </label>

      <AnimatePresence>
        <motion.p
          v-if="busy || freeTextSubmission"
          class="guided-composer-status-line guided-composer-submission"
          :class="freeTextSubmission ? `is-${freeTextSubmission.status}` : 'is-busy'"
          :role="freeTextSubmission?.status === 'failed' ? 'alert' : 'status'"
          aria-live="polite"
          :initial="reduceMotion ? false : { opacity: 0, y: -3 }"
          :animate="{ opacity: 1, y: 0 }"
          :exit="{ opacity: 0, y: -3 }"
          :transition="{ duration: reduceMotion ? 0 : 0.16 }"
        >
          <span v-if="busy" class="guided-composer-busy">
            <span class="guided-composer-status-dot" aria-hidden="true" />
            助手处理中
          </span>
          <span
            v-if="freeTextSubmission"
            class="guided-composer-submitted-text"
            :title="freeTextSubmission.text"
          >
            你说：“{{ freeTextSubmission.text }}”
          </span>
          <span
            v-if="freeTextSubmission"
            class="guided-composer-status"
            :title="composerStatusText"
          >
            {{ composerStatusText }}
          </span>
        </motion.p>
      </AnimatePresence>

      <div class="guided-composer-row">
        <div class="guided-composer-field">
          <span class="guided-composer-field-icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-4-4" />
            </svg>
          </span>
          <input
            id="guided-conversation-input"
            ref="searchInputRef"
            v-model="composerText"
            type="text"
            autocomplete="off"
            enterkeyhint="send"
            placeholder="如：河边、有山地、红砖建筑，也可补充店名或路牌"
            :disabled="busy"
            aria-describedby="guided-composer-hint"
            @compositionstart="composing = true"
            @compositionend="composing = false"
            @keydown.enter="onComposerEnter"
          />
        </div>
        <button
          type="submit"
          class="primary-btn guided-composer-submit"
          :disabled="!canSendComposer"
          :title="busy ? '当前操作完成后可继续发送' : '发送当前线索（Enter）'"
        >
          <span v-if="busy" class="guided-composer-spinner" aria-hidden="true" />
          <span>{{ applying ? '发送中…' : busy ? '处理中…' : '发送线索' }}</span>
          <svg
            v-if="!busy"
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            aria-hidden="true"
          >
            <path d="m5 12 14-7-4 14-3-6z" />
            <path d="m12 13 7-8" />
          </svg>
        </button>
      </div>

      <div class="guided-composer-foot">
        <p id="guided-composer-hint" class="guided-composer-hint">{{ composerHint }}</p>
        <span class="guided-composer-enter-hint" aria-hidden="true"><kbd>Enter</kbd> 发送</span>
      </div>
    </form>
  </section>
</template><style scoped>
.guided-dialogue-shell {
  --guided-ink: var(--ink);
  --guided-muted: var(--muted);
  --guided-line: var(--line);
  --guided-line-strong: var(--line-strong);
  --guided-teal: var(--amber);
  --guided-teal-dark: var(--amber-bright);
  --guided-teal-soft: var(--amber-soft);
  --guided-warn: #e0b26a;
  --guided-danger: var(--coral);
  overflow: hidden;
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 0;
  min-width: 0;
  min-height: 0;
  border: 0;
  border-radius: 0;
  color: var(--guided-ink);
  background: transparent;
  box-shadow: none;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-panel {
  display: block;
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
  scroll-behavior: smooth;
  scroll-padding-top: 8px;
  padding: 0;
  background: transparent;
}

.guided-dock-header {
  position: relative;
  z-index: auto;
  padding: 0;
  border-bottom: 1px solid var(--guided-line);
  background: transparent;
}

.guided-dock-details {
  margin: 0;
  border: 0;
  background: transparent;
}

.guided-dock-summary {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  margin: 0;
  padding: 6px 16px;
  color: var(--guided-muted);
  list-style: none;
  cursor: pointer;
}

.guided-dock-summary::-webkit-details-marker {
  display: none;
}

.guided-dock-summary:hover h2 {
  color: var(--guided-teal);
}

.guided-dock-summary h2 {
  overflow: hidden;
  margin: 0;
  color: var(--guided-ink);
  font-size: 15px;
  font-weight: 700;
  line-height: 22px;
  letter-spacing: -0.005em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.guided-progress {
  flex-shrink: 0;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  white-space: nowrap;
}

.guided-details-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding-left: 8px;
  border-left: 1px solid var(--guided-line);
  color: var(--guided-teal);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  white-space: nowrap;
}

.guided-details-toggle::after {
  content: "▾";
  color: var(--faint);
  font-size: 10px;
  line-height: 1;
  transition: transform 0.16s ease;
}

.guided-dock-details[open] .guided-details-toggle::after {
  transform: rotate(180deg);
}

.guided-dock-details-body {
  display: grid;
  gap: 9px;
  padding: 8px 16px 14px;
  border-top: 1px solid var(--guided-line);
  background: transparent;
}

.guided-details-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
}

.guided-details-heading [role="status"] {
  overflow-wrap: anywhere;
  color: var(--guided-muted);
  text-align: right;
}

.guided-details-backtrack {
  display: flex;
}

.guided-back-btn {
  min-height: 30px;
  padding: 4px 8px;
  border-color: var(--guided-line);
  border-radius: 4px;
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-back-btn:hover:not(:disabled) {
  border-color: var(--amber-line);
  color: var(--guided-teal);
  background: var(--guided-teal-soft);
}

.guided-path {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.guided-path li {
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  min-width: 0;
  min-height: 30px;
  margin: 0;
  padding: 4px 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  opacity: 1;
}

.guided-path li + li {
  border-top: 1px solid var(--guided-line);
}

.guided-path li.is-pending .guided-step-main strong {
  color: var(--faint);
}

.guided-step-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: 1px solid var(--guided-line-strong);
  border-radius: 3px;
  color: var(--guided-muted);
  background: transparent;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 400;
  line-height: 18px;
}

.guided-path li.is-done .guided-step-index {
  border-color: var(--sage-line);
  color: var(--sage);
  background: var(--sage-soft);
}

.guided-path li.is-current .guided-step-index {
  border-color: var(--guided-teal);
  color: var(--on-amber);
  background: var(--guided-teal);
}

.guided-step-main {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}

.guided-step-label {
  color: var(--faint);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
  white-space: nowrap;
}

.guided-step-main strong {
  overflow: hidden;
  color: var(--guided-ink);
  font-size: 13px;
  font-weight: 600;
  line-height: 18px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.guided-backtrack-btn,
.guided-link-btn {
  flex: 0 0 auto;
  min-height: 20px;
  padding: 1px 2px;
  border: 0;
  border-radius: 3px;
  color: var(--guided-teal);
  background: transparent;
  font-size: 13px;
  font-weight: 400;
  line-height: 18px;
  cursor: pointer;
}

.guided-backtrack-btn:hover:not(:disabled),
.guided-link-btn:hover:not(:disabled) {
  color: var(--guided-teal-dark);
  text-decoration: underline;
  text-underline-offset: 3px;
}

.guided-backtrack-btn:disabled,
.guided-link-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.guided-task-message {
  margin: 0;
  padding: 8px 0 0;
  border-top: 1px solid var(--guided-line);
  background: transparent;
}

.guided-task-message-label {
  display: block;
  color: var(--amber);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
  letter-spacing: 0.04em;
}

.guided-task-message p {
  margin: 2px 0 0;
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 19px;
  overflow-wrap: anywhere;
}

.guided-backtrack-confirm {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--coral-line);
  border-radius: 4px;
  color: var(--guided-danger);
  background: var(--coral-soft);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-backtrack-confirm > div {
  display: flex;
  flex: 0 0 auto;
  gap: 5px;
}

.guided-backtrack-confirm .ghost-btn,
.guided-backtrack-confirm .mini-btn {
  min-height: 32px;
  padding: 6px 10px;
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-main {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.guided-error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--coral-line);
  border-radius: 5px;
  background: var(--coral-soft);
}

.guided-error p {
  color: var(--guided-danger);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-error .ghost-btn {
  min-height: 36px;
  padding: 8px 12px;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-loading {
  padding: 20px 12px;
  color: var(--guided-muted);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  text-align: center;
}

.guided-receipt {
  justify-self: start;
  padding: 3px 0 3px 8px;
  border-left: 2px solid var(--amber-line);
  color: var(--guided-muted);
  background: transparent;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-resolved {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 16px;
  border: 1px solid var(--sage-line);
  border-radius: 7px;
  background: var(--sage-soft);
}

.guided-resolved-label {
  color: var(--sage);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  letter-spacing: 0.1em;
}

.guided-resolved strong {
  color: var(--guided-ink);
  font-size: 18px;
  font-weight: 700;
  line-height: 28px;
  letter-spacing: -0.01em;
}

.guided-resolved p {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-confidence-result {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  border: 1px solid var(--guided-line);
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.018);
  box-shadow: none;
}

.guided-confidence-result.is-candidates {
  border-color: var(--amber-line);
}

.guided-confidence-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 9px;
}

.guided-confidence-head > div {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.guided-confidence-kicker {
  color: var(--guided-teal);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  letter-spacing: 0.08em;
}

.guided-confidence-result.is-candidates .guided-confidence-kicker {
  color: var(--guided-warn);
}

.guided-confidence-head strong {
  color: var(--guided-ink);
  font-size: 15px;
  font-weight: 600;
  line-height: 22px;
}

.guided-confidence-note {
  flex: 0 0 auto;
  padding: 1px 5px;
  border: 1px solid var(--guided-line);
  border-radius: 3px;
  color: var(--faint);
  background: transparent;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  white-space: nowrap;
}

.guided-confidence-explanation {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-confidence-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px dashed var(--guided-line);
}

.guided-confidence-actions .ghost-btn {
  flex: 0 0 auto;
  min-height: 36px;
  padding: 8px 12px;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  white-space: nowrap;
}

.guided-confidence-actions span {
  color: var(--faint);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-question {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  border: 1px solid var(--guided-line);
  border-left: 2px solid var(--amber);
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.018);
  box-shadow: none;
}

.guided-review-box {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.guided-low-confidence-note {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  border: 1px solid var(--amber-line);
  border-radius: 5px;
  background: var(--amber-soft);
}

.guided-low-confidence-note strong {
  color: var(--guided-warn);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-low-confidence-note span {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.guided-section-title {
  color: var(--guided-ink);
  font-size: 15px;
  font-weight: 600;
  line-height: 22px;
}

.guided-one-question,
.guided-section-note {
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-review-progress {
  flex-shrink: 0;
  color: var(--guided-teal);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
}

.guided-question-message {
  color: var(--guided-ink);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-question-reason,
.guided-section-hint {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-question-clue {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  border: 1px dashed var(--guided-line-strong);
  border-radius: 5px;
  background: var(--panel-soft);
}

.guided-question-clue strong {
  color: var(--guided-ink);
  font-size: 15px;
  font-weight: 600;
  line-height: 22px;
}

.guided-question-clue > span,
.guided-question-hint {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-clue-new {
  justify-self: start;
  color: var(--guided-teal) !important;
  font-family: var(--font-mono);
  font-size: 13px !important;
  line-height: 20px;
  font-weight: 400;
}

.guided-inline-actions,
.guided-answer-options {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.guided-inline-actions .mini-btn,
.guided-answer-options .option-chip {
  min-height: 36px;
  padding: 8px 12px;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-unknown-btn {
  align-self: flex-start;
  min-height: 36px;
  padding: 8px 12px;
  border: 1px dashed var(--guided-line-strong);
  border-radius: 3px;
  color: var(--guided-muted);
  background: transparent;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  cursor: pointer;
}

.guided-unknown-btn:hover:not(:disabled) {
  border-color: var(--guided-teal);
  color: var(--guided-teal);
}

.guided-unknown-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.guided-review-nav {
  display: flex;
  justify-content: flex-start;
}

.guided-review-nav .ghost-btn,
.guided-candidates-toolbar .ghost-btn {
  min-height: 36px;
  padding: 8px 12px;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-next-step {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px;
  border: 1px solid var(--sage-line);
  border-left: 2px solid var(--sage);
  border-radius: 7px;
  background: var(--sage-soft);
}

.guided-next-step > div {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.guided-next-step strong {
  color: var(--sage);
  font-size: 15px;
  font-weight: 600;
  line-height: 22px;
}

.guided-next-step span {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-next-step .primary-btn {
  flex: 0 0 auto;
  min-height: 40px;
  padding: 10px 14px;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidates,
.guided-exclusions {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding-top: 9px;
  border-top: 1px solid var(--guided-line);
}

.guided-candidates-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.guided-empty {
  padding: 16px;
  border: 1px dashed var(--guided-line-strong);
  border-radius: 5px;
  color: var(--guided-muted);
  background: transparent;
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidate-card {
  display: grid;
  flex: 0 0 auto;
  gap: 8px;
  align-items: stretch;
  justify-content: initial;
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--guided-line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.018);
  cursor: default;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.guided-candidate-card:hover {
  border-color: var(--guided-line-strong);
  background: rgba(255, 255, 255, 0.03);
}

.guided-candidate-card.selected {
  border-color: var(--amber-line);
  background: var(--amber-soft);
  box-shadow: none;
}

.guided-candidate-summary {
  display: flex;
  align-items: flex-start;
  min-width: 0;
}

.guided-candidate-select {
  display: flex;
  flex: 1 1 auto;
  align-items: flex-start;
  gap: 8px;
  min-width: 0;
  padding: 1px;
  border: 0;
  border-radius: 3px;
  color: inherit;
  background: transparent;
  text-align: left;
  cursor: pointer;
}

.guided-candidate-select:hover:not(:disabled) .guided-candidate-main strong {
  color: var(--guided-teal);
}

.guided-candidate-select:disabled {
  cursor: not-allowed;
}

.guided-candidate-number {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--guided-line-strong);
  border-radius: 3px;
  color: var(--guided-muted);
  background: transparent;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidate-card.selected .guided-candidate-number {
  border-color: var(--guided-teal);
  color: var(--on-amber);
  background: var(--guided-teal);
}

.guided-candidate-main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.guided-candidate-main strong {
  overflow: hidden;
  color: var(--guided-ink);
  font-size: 15px;
  font-weight: 600;
  line-height: 22px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.guided-candidate-name-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 7px;
  min-width: 0;
}

.guided-candidate-meta,
.guided-candidate-point {
  color: var(--faint);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidate-meta {
  display: -webkit-box;
  overflow: hidden;
  min-width: 0;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
}

.guided-candidate-point {
  flex: 0 0 auto;
  color: var(--blue);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidate-point.missing {
  color: var(--faint);
}

.guided-candidate-detail {
  margin-left: 36px;
  border-top: 1px dashed var(--guided-line);
}

.guided-candidate-detail summary {
  min-height: 32px;
  padding: 6px 0 2px;
  color: var(--guided-teal);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  cursor: pointer;
  list-style: none;
}

.guided-candidate-detail summary:hover {
  color: var(--guided-teal-dark);
  text-decoration: underline;
  text-underline-offset: 3px;
}

.guided-candidate-detail-body {
  display: grid;
  gap: 4px;
  padding: 2px 0 1px;
}

.guided-candidate-reason,
.guided-candidate-point-note {
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  overflow-wrap: anywhere;
}

.guided-candidate-point-note {
  color: var(--guided-warn);
}

.guided-candidate-detail :deep(.place-evidence) {
  margin-top: 0;
  color: var(--guided-muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-candidate-actions {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  padding-left: 36px;
}

.guided-candidate-actions .mini-btn {
  min-height: 36px;
  padding: 8px 12px;
  border-radius: 5px;
  font-size: 14px;
  line-height: 20px;
  font-weight: 400;
}

.guided-candidate-actions .mini-btn.confirm {
  border-color: var(--guided-teal);
  color: var(--on-amber);
  background: var(--guided-teal);
}

.guided-candidate-actions .mini-btn.confirm:hover:not(:disabled) {
  border-color: var(--guided-teal-dark);
  background: var(--guided-teal-dark);
}

.guided-candidate-actions .mini-btn.reject {
  border-color: var(--coral-line);
  color: var(--guided-danger);
  background: transparent;
}

.guided-candidate-actions .mini-btn.reject:hover:not(:disabled) {
  border-color: var(--guided-danger);
  color: var(--bg);
  background: var(--guided-danger);
}

.guided-candidate-actions .mini-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.guided-exclusion-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 12px;
  border: 1px solid var(--coral-line);
  border-radius: 4px;
  color: var(--guided-danger);
  background: var(--coral-soft);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-composer {
  position: relative;
  z-index: 30;
  display: grid;
  flex: 0 0 auto;
  gap: 8px;
  min-width: 0;
  padding: 12px 16px;
  border-top: 1px solid var(--guided-line);
  background: rgba(13, 21, 25, 0.985);
  box-shadow: 0 -10px 26px rgba(0, 0, 0, 0.24);
  transition: border-color 0.16s ease;
}

.guided-composer.is-busy {
  border-top-color: var(--amber-line);
}

.guided-composer-status-line {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 2px 6px;
  min-width: 0;
  margin: 0;
  padding: 6px 8px;
  border: 1px solid var(--guided-line);
  border-radius: 4px;
  color: var(--guided-muted);
  background: rgba(255, 255, 255, 0.018);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-composer-status-line.is-failed {
  border-color: var(--coral-line);
  color: var(--guided-danger);
  background: var(--coral-soft);
}

.guided-composer-status-line.is-unconfirmed {
  border-color: var(--amber-line);
  color: var(--guided-warn);
  background: var(--amber-soft);
}

.guided-composer-busy {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 5px;
  color: var(--guided-teal);
  font-weight: 500;
}

.guided-composer-status-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
  animation: guided-composer-pulse 1.1s ease-in-out infinite;
}

.guided-composer-submitted-text {
  overflow: hidden;
  flex: 0 1 auto;
  max-width: 100%;
  color: var(--ink-soft);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.guided-composer-status {
  flex: 1 1 180px;
  min-width: 0;
  color: currentColor;
}

.guided-composer-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
  align-items: stretch;
}

.guided-composer-field {
  position: relative;
  display: flex;
  align-items: center;
  min-width: 0;
  border-radius: 6px;
}

.guided-composer-field-icon {
  position: absolute;
  left: 12px;
  z-index: 1;
  display: inline-flex;
  color: var(--faint);
  pointer-events: none;
}

.guided-composer-row .guided-composer-field input {
  width: 100%;
  min-height: 42px;
  padding: 10px 12px 10px 34px;
  border: 1px solid var(--guided-line-strong);
  border-radius: 6px;
  color: var(--guided-ink);
  background: rgba(255, 255, 255, 0.028);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  outline: none;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.guided-composer-row .guided-composer-field input::placeholder {
  color: var(--faint);
}

.guided-composer-field:focus-within {
  outline: none;
  box-shadow: 0 0 0 1px var(--amber-line);
  border-radius: 6px;
}

.guided-composer-row .guided-composer-field input:focus {
  border-color: var(--amber);
}

.guided-composer-row .guided-composer-field input:focus-visible {
  outline: none;
}

.guided-composer-row .guided-composer-field input:disabled {
  color: var(--guided-muted);
  background: rgba(255, 255, 255, 0.014);
  cursor: not-allowed;
}

.guided-composer-row .guided-composer-submit {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  min-height: 42px;
  padding: 0 14px;
  border: 1px solid var(--amber);
  border-radius: 6px;
  color: var(--on-amber);
  background: var(--amber);
  box-shadow: none;
  font-size: 14px;
  line-height: 20px;
  font-weight: 400;
  white-space: nowrap;
}

.guided-composer-row .guided-composer-submit:hover:not(:disabled) {
  border-color: var(--amber-bright);
  background: var(--amber-bright);
}

.guided-composer-row .guided-composer-submit:disabled {
  border-color: var(--guided-line);
  color: var(--faint);
  background: rgba(255, 255, 255, 0.03);
  box-shadow: none;
}

.guided-composer-spinner {
  width: 12px;
  height: 12px;
  border: 1.5px solid rgba(23, 16, 6, 0.3);
  border-top-color: var(--on-amber);
  border-radius: 50%;
  animation: guided-composer-spin 0.75s linear infinite;
}

.guided-composer-foot {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.guided-composer-hint {
  display: -webkit-box;
  overflow: hidden;
  flex: 1 1 auto;
  min-width: 0;
  color: var(--faint);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
}

.guided-composer-enter-hint {
  flex: 0 0 auto;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  white-space: nowrap;
}

.guided-composer-enter-hint kbd {
  padding: 1px 4px;
  border: 1px solid var(--guided-line-strong);
  border-bottom-width: 2px;
  border-radius: 3px;
  color: var(--guided-muted);
  background: rgba(255, 255, 255, 0.03);
  font: inherit;
}

.guided-dialogue-shell button:focus-visible,
.guided-dialogue-shell summary:focus-visible {
  outline: 1.5px solid var(--guided-teal);
  outline-offset: 2px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes guided-composer-pulse {
  0%,
  100% {
    opacity: 0.4;
    transform: scale(0.9);
  }
  50% {
    opacity: 1;
    transform: scale(1.1);
  }
}

@keyframes guided-composer-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 480px) {
  .guided-dialogue-shell {
    border-radius: 0;
  }

  .guided-dock-summary {
    gap: 6px;
    min-height: 42px;
    padding: 5px 10px;
  }

  .guided-dock-summary h2 {
    font-size: 14px;
    line-height: 20px;
  }

  .guided-progress {
    padding-inline: 0;
    font-size: 12px;
  }

  .guided-details-toggle {
    gap: 3px;
    padding-left: 6px;
    font-size: 12px;
  }

  .guided-dock-details-body {
    gap: 8px;
    padding: 7px 10px 12px;
  }

  .guided-step-main {
    gap: 6px;
  }

  .guided-main {
    gap: 8px;
    padding: 8px 10px 12px;
  }

  .guided-question,
  .guided-confidence-result {
    padding: 9px;
  }

  .guided-next-step {
    align-items: flex-start;
    flex-direction: column;
  }

  .guided-next-step .primary-btn {
    align-self: stretch;
  }

  .guided-candidate-detail,
  .guided-candidate-actions {
    margin-left: 0;
    padding-left: 0;
  }

  .guided-candidate-actions {
    justify-content: stretch;
  }

  .guided-candidate-actions .mini-btn {
    flex: 1 1 0;
  }

  .guided-composer {
    padding: 7px 10px 8px;
  }

  .guided-composer-enter-hint {
    display: none;
  }
}

@media (max-width: 360px) {
  .guided-dock-summary {
    gap: 5px;
    padding-inline: 8px;
  }

  .guided-step-main {
    grid-template-columns: minmax(0, 1fr);
    gap: 0;
  }

  .guided-backtrack-btn {
    padding-inline: 0;
  }

  .guided-composer-row {
    grid-template-columns: minmax(0, 1fr);
  }

  .guided-composer-row .guided-composer-submit {
    width: 100%;
  }
}

@media (hover: none), (pointer: coarse) {
  .guided-composer-enter-hint {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .guided-panel {
    scroll-behavior: auto;
  }

  .guided-candidate-card,
  .guided-composer,
  .guided-composer-row .guided-composer-field input {
    transition: none;
  }

  .guided-composer-status-dot,
  .guided-composer-spinner {
    animation: none;
  }
}
</style>

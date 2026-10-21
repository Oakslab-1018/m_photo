<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
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
    ? '模型给出了首选国家，请结合照片确认后再进入下一层级。'
    : '以下候选较为接近，请结合照片明确选择。'
})
const shortcutTitle = computed(() => {
  if (countryShortcut.value === 'high') return '推荐国家 / 地区'
  if (props.confidence?.tier === 'low') return '候选仅供参考'
  return '国家 / 地区候选待确认'
})
const shortcutNote = computed(() => {
  if (countryShortcut.value === 'high') return '可继续深化城市'
  if (props.confidence?.tier === 'low') return '仅在能够确认时选择'
  return '请确认最符合照片的候选'
})
const shortcutFooter = computed(() => {
  if (countryShortcut.value === 'high') {
    return '这是当前国家预测；如需继续深化城市，可确认该国家后进入下一步。'
  }
  if (props.confidence?.tier === 'low') {
    return '这些只是线索不足时的可选候选；仅在能够确认时选择，未确认前不会推进。'
  }
  return '请从以上候选中明确确认国家；确认后会进入下一级，未确认前不会推进。'
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

const progressText = computed(() => {
  if (isResolved.value) return '2 / 2 · 已完成'
  const current = GUIDED_STEPS.findIndex((step) => step.level === stage.value)
  return `${Math.max(1, current + 1)} / ${GUIDED_STEPS.length} · 当前步骤`
})

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
const candidateSectionNote = computed(() => {
  const composition = candidateComposition.value
  if (composition.hasUserText) {
    return composition.mixed ? '线索匹配优先 · 其余按排序分' : '线索匹配候选'
  }
  if (composition.hasCatalogSearch) {
    return composition.mixed ? '目录检索优先 · 其余按排序分' : '目录检索候选'
  }
  if (composition.hasModelScores) return '候选可信度前五'
  return `目录候选（当前显示 ${topCandidates.value.length} 项）`
})
const candidateSectionHint = computed(() => {
  const composition = candidateComposition.value
  if (!props.locationConfirmationReady) {
    return composition.hasModelScores && !composition.hasExplicitSelectionBasis
      ? '新增观察待核对，排名仍可查看，但暂不能确认或排除地点。'
      : '新增观察待核对，候选仍可查看，但暂不能确认或排除地点。'
  }
  if (composition.hasUserText) {
    return composition.mixed
      ? '带“线索匹配候选”的条目来自你提供的文字线索，优先展示且不是模型概率；其余条目保留真实排序分。'
      : '带“线索匹配候选”的条目来自你提供的文字线索，优先展示且不是模型概率。'
  }
  if (composition.hasCatalogSearch) {
    return composition.mixed
      ? '目录检索命中优先展示且不是模型概率；其余条目保留真实排序分。'
      : '目录检索命中优先展示，不是模型概率。'
  }
  if (composition.hasModelScores) {
    return '候选可信度前五 · 排序分仅供参考，不代表概率'
  }
  return `按目录顺序展示，可输入${candidateLevelName.value}名称查找更多；尚无图像评分，不代表最可能的五个`
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
const composerAssistantMessage = computed(() => {
  if (isResolved.value) {
    return safeGuidedText(props.action?.message).trim()
      || '定位已完成；如发现新线索，仍可继续补充。'
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
    return '定位已完成；新线索会交回助手重新核对，不会自动改动已确认结果。'
  }
  if (!props.candidatesUnlocked) {
    return '请描述照片中可见的招牌、道路方向、语言或地名；提交不会自动确认地点。'
  }
  return `可描述照片中可见的线索，也可直接输入${candidateLevelName.value}名称检索目录；提交不会自动确认地点。`
})
const composerReceiptText = computed(() => safeGuidedReceipt(props.receipt?.summary))
const receiptStatusText = computed(() => {
  const summary = composerReceiptText.value
  if (summary) return summary
  const status = props.receipt?.status?.trim().toLowerCase()
  if (status === 'unchanged') return '这条线索此前已保存，无需重复提交。'
  if (status === 'unrecognized') return '没有识别到新的线索。'
  return '操作已应用'
})
const composerStatusText = computed(() => {
  const submission = props.freeTextSubmission
  if (!submission) return ''
  if (submission.status === 'pending') return '正在发送这条线索，等待服务器确认…'
  if (submission.status === 'failed') {
    const reason = safeGuidedText(submission.error).trim()
    return `${reason || '提交失败'} · 草稿已保留，可修改后再次发送。`
  }
  if (submission.status === 'unconfirmed') {
    const reason = safeGuidedText(submission.error).trim()
    return `${reason || '服务器未确认此次提交'} · 草稿已保留，请核对最新状态后重试。`
  }
  if (submission.status === 'unchanged') {
    return '这条线索已记录，无需重复添加。'
  }
  return composerReceiptText.value
    ? `助手反馈：${composerReceiptText.value}`
    : '已发送，助手会结合这条线索更新判断。'
})

/** Scroll only the guided panel; never scroll the document or a candidate card. */
function scrollPanelTo(target: HTMLElement | null): void {
  const panel = panelRef.value
  if (!panel || !target) return
  const panelRect = panel.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const nextTop = panel.scrollTop + targetRect.top - panelRect.top
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
    <div ref="panelRef" class="guided-panel" aria-live="polite">
    <div class="guided-panel-head">
      <div>
        <span class="guided-kicker">引导定位</span>
        <h2>{{ stageTitle }}</h2>
      </div>
      <div class="guided-panel-head-actions">
        <span class="guided-progress">{{ progressText }}</span>
        <button
          v-if="previousLevel && dialogue"
          type="button"
          class="ghost-btn guided-back-btn"
          :disabled="busy"
          :title="`回到上一层级：${locationLevelLabel(previousLevel)}（保留其他证据与排除）`"
          @click="emit('backtrack', previousLevel)"
        >
          返回上一步：{{ locationLevelLabel(previousLevel) }}
        </button>
      </div>
    </div>

    <div v-if="displayError" class="guided-error" role="alert">
      <p>{{ displayError }}</p>
      <button type="button" class="ghost-btn" :disabled="busy" @click="requestRetry">
        {{ canRetry ? '重试上次操作' : '重新加载' }}
      </button>
    </div>

    <div v-if="loading && !dialogue" class="guided-loading">
      正在读取服务器保存的引导状态…
    </div>

    <template v-else-if="dialogue">
      <div v-if="receipt?.summary || receipt?.status" class="guided-receipt" role="status">
        {{ receiptStatusText }}
      </div>

      <div v-if="isResolved" class="guided-resolved">
        <span class="guided-resolved-label">最终结果</span>
        <strong>{{ targetLabel(dialogue.confirmed.city) }}</strong>
        <p>已根据您的确认完成定位；城市坐标仅为代表点，不是照片 GPS。</p>
      </div>

      <ol class="guided-path" aria-label="已确认路径">
        <li
          v-for="(step, index) in pathSteps"
          :key="step.level"
          :class="`is-${step.state}`"
        >
          <span class="guided-step-index">{{ index + 1 }}</span>
          <div class="guided-step-main">
            <span class="guided-step-label">{{ step.label }}</span>
            <strong>{{ step.target ? targetLabel(step.target) : '待确认' }}</strong>
          </div>
          <button
            v-if="step.target && !busy"
            type="button"
            class="guided-backtrack-btn"
            :title="`返回${step.label}会清除后续确认`"
            @click="requestBacktrack(step.level)"
          >
            返回此步
          </button>
        </li>
      </ol>

      <div v-if="backtrackPrompt" class="guided-backtrack-confirm" role="alert">
        <span>返回此步会清除它后面已经确认的层级，确定继续？</span>
        <div>
          <button type="button" class="ghost-btn" :disabled="busy" @click="backtrackPrompt = null">取消</button>
          <button type="button" class="mini-btn reject" :disabled="busy" @click="confirmBacktrack">确认返回</button>
        </div>
      </div>

      <section
        v-if="!isResolved && countryShortcut"
        class="guided-confidence-result"
        :class="'is-' + countryShortcut"
      >
        <div class="guided-confidence-head">
          <div>
            <span class="guided-confidence-kicker">国家 / 地区预测</span>
            <strong>{{ shortcutTitle }}</strong>
          </div>
          <span class="guided-confidence-note">{{ shortcutNote }}</span>
        </div>
        <p class="guided-confidence-explanation">{{ shortcutExplanation }}</p>

        <article
          v-for="candidate in shortcutCandidates"
          :key="candidateTargetKey(candidate.target)"
          class="guided-candidate-card guided-confidence-card"
          :class="{ selected: isCandidateSelected(candidate.target) }"
          :tabindex="busy ? -1 : 0"
          :aria-current="isCandidateSelected(candidate.target) ? 'true' : undefined"
          @click="selectCandidate(candidate.target)"
          @keydown.enter.prevent="selectCandidate(candidate.target)"
          @keydown.space.prevent="selectCandidate(candidate.target)"
        >
          <div class="guided-candidate-main">
            <strong>{{ targetLabel(candidate.target) }}</strong>
            <span class="guided-candidate-meta">{{ candidateScoreLabel(candidate) }}</span>
            <span v-if="candidatePointLabel(candidate.target)" class="guided-candidate-point">
              {{ candidatePointLabel(candidate.target) }}
            </span>
            <p v-if="candidate.reason" class="guided-candidate-reason">{{ candidate.reason }}</p>
            <PlaceMatchEvidence v-if="candidate.place_match" :match="candidate.place_match" />
          </div>
          <div class="guided-candidate-actions">
            <button
              type="button"
              class="mini-btn confirm"
              :disabled="busy || !locationConfirmationReady"
              @click.stop="emit('confirm-location', candidate.target)"
            >
              确认{{ locationLevelLabel(candidate.target.level) }}
            </button>
            <button
              type="button"
              class="mini-btn reject"
              :disabled="busy || !locationConfirmationReady"
              @click.stop="emit('reject-location', candidate.target)"
            >
              排除{{ locationLevelLabel(candidate.target.level) }}
            </button>
          </div>
        </article>

        <div v-if="!shortcutCandidates.length" class="guided-empty">
          服务器候选暂未就绪，请重新加载；页面不会自行拼装地点。
        </div>
        <div class="guided-confidence-actions">
          <button
            v-if="countryShortcut === 'high'"
            type="button"
            class="ghost-btn"
            :disabled="busy"
            @click="focusFreeText"
          >
            补充线索细化
          </button>
          <span>{{ shortcutFooter }}</span>
        </div>
      </section>

      <section
        v-if="!isResolved && !countryShortcut"
        ref="questionSectionRef"
        class="guided-question"
      >
        <template v-if="phase === 'review'">
          <div v-if="showLowCluePrompt" class="guided-low-confidence-note">
            <strong>当前信息不足，先补充可观察线索</strong>
            <span>请回答下面的问题，或在下方输入文字线索；提交后会重新比较国家候选，不会自动确认地点。</span>
          </div>
          <div v-if="isReviewClueQuestion" class="guided-review-box">
            <div class="guided-section-head">
              <span class="guided-section-title">核对观察</span>
              <span v-if="reviewProgress" class="guided-review-progress">
                第 {{ reviewProgress.current }} / {{ reviewProgress.total }}
              </span>
              <span v-else class="guided-one-question">本轮仅此一条观察</span>
            </div>
            <template v-if="currentReviewClue">
              <p class="guided-question-message">{{ questionMessage }}</p>
              <p class="guided-question-reason">
                {{ questionReason }}
              </p>
              <div class="guided-question-clue">
                <strong>{{ reviewedClueText(currentReviewClue) }}</strong>
                <span>{{ currentReviewClue.description || '未提供详细说明' }}</span>
                <span v-if="isRecentClue(currentReviewClue.id)" class="guided-clue-new">新线索</span>
              </div>
              <div class="guided-inline-actions">
                <button
                  type="button"
                  class="mini-btn confirm"
                  :disabled="busy"
                  @click="emit('confirm-clue', currentReviewClue.id)"
                >
                  判断准确
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
                不确定 / 无法判断（不代表排除）
              </button>
            </template>
            <div v-else class="guided-question-hint">
              当前观察暂时无法显示，请重新加载后再试。
            </div>
          </div>

          <div v-else-if="blockingQuestion" class="guided-review-box">
            <div class="guided-section-head">
              <span class="guided-section-title">{{ guidedQuestionTitle(blockingQuestion) }}</span>
              <span class="guided-one-question">本轮仅此一问</span>
            </div>
            <p v-if="narratedLowQuestion" class="guided-question-reason">
              <strong>问题说明：</strong>{{ narratedLowQuestion }}
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
              请在下方输入可观察线索；自由文本不会直接推进确认层级。
            </div>
            <button
              type="button"
              class="guided-unknown-btn"
              :disabled="busy"
              @click="emit('unknown')"
            >
              不确定 / 跳过此问题
            </button>
          </div>

          <div v-else-if="pendingClueCount > 0" class="guided-question-hint">
            还有 {{ pendingClueCount }} 条观察待核对，请重新加载后再试。
          </div>

          <div v-else class="guided-next-step" role="status">
            <div>
              <strong>观察核对已完成</strong>
              <span>下一步查看地点候选；选择符合照片的地点后再确认。</span>
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
              返回上一条观察（重新核对）
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
            重审上一条观察
          </button>
        </div>

        <div v-if="pendingQuestion?.kind === 'confirm_location'" class="guided-question-hint">
          请在下方候选中明确确认当前层级；不会根据排序第一名自动确认。
        </div>

        <p class="guided-section-hint">{{ candidateSectionHint }}</p>
        <div v-if="!topCandidates.length" class="guided-empty">
          当前层级暂无候选。请在下方补充照片中可见的招牌、地名或文字；补充线索不会自动确认地点。
        </div>
        <article
          v-for="candidate in topCandidates"
          :key="candidateTargetKey(candidate.target)"
          class="guided-candidate-card"
          :class="{ selected: isCandidateSelected(candidate.target) }"
          :tabindex="busy ? -1 : 0"
          :aria-current="isCandidateSelected(candidate.target) ? 'true' : undefined"
          @click="selectCandidate(candidate.target)"
          @keydown.enter.prevent="selectCandidate(candidate.target)"
          @keydown.space.prevent="selectCandidate(candidate.target)"
        >
          <div class="guided-candidate-main">
            <strong>{{ targetLabel(candidate.target) }}</strong>
            <span class="guided-candidate-meta">{{ candidateScoreLabel(candidate) }}</span>
            <span v-if="candidatePointLabel(candidate.target)" class="guided-candidate-point">
              {{ candidatePointLabel(candidate.target) }}
            </span>
            <span v-else class="guided-candidate-point missing">
              此候选无坐标，地图不放置候选标记；选中时仅聚焦已确认父级范围
            </span>
            <p v-if="candidate.reason" class="guided-candidate-reason">{{ candidate.reason }}</p>
            <PlaceMatchEvidence v-if="candidate.place_match" :match="candidate.place_match" />
          </div>
          <div class="guided-candidate-actions">
            <button
              type="button"
              class="mini-btn confirm"
              :disabled="busy || !locationConfirmationReady"
              @click.stop="emit('confirm-location', candidate.target)"
            >
              确认{{ locationLevelLabel(candidate.target.level) }}
            </button>
            <button
              type="button"
              class="mini-btn reject"
              :disabled="busy || !locationConfirmationReady"
              @click.stop="emit('reject-location', candidate.target)"
            >
              排除{{ locationLevelLabel(candidate.target.level) }}
            </button>
          </div>
        </article>
      </section>

      <section v-if="candidatesPhase && dialogue.exclusions.length" class="guided-exclusions">
        <div class="guided-section-head">
          <span class="guided-section-title">已排除地点</span>
          <span class="guided-section-note">可撤回</span>
        </div>
        <p class="guided-section-hint">地点排除只作用于对应目标，不会影响同名地点或其他层级。</p>
        <div v-for="target in dialogue.exclusions" :key="`${target.level}-${target.id}`" class="guided-exclusion-row">
          <span>{{ locationLevelLabel(target.level) }} · {{ targetLabel(target) }}</span>
          <button type="button" class="guided-link-btn" :disabled="busy" @click="emit('retract-target', target)">
            撤回排除
          </button>
        </div>
      </section>

    </template>

    <div v-else-if="!loading" class="guided-loading">
      引导状态尚未就绪，请重试。
    </div>
    </div>

    <form
      v-if="dialogue"
      class="guided-composer"
      aria-labelledby="guided-composer-label"
      @submit.prevent="submitComposer"
    >
      <div class="guided-composer-head">
        <label id="guided-composer-label" for="guided-conversation-input">
          补充线索，与定位助手对话
        </label>
        <span>对话输入 · 也支持地名检索</span>
      </div>

      <p
        v-if="composerAssistantMessage"
        class="guided-composer-assistant"
        :title="composerAssistantMessage"
      >
        <strong>助手</strong>
        <span>{{ composerAssistantMessage }}</span>
      </p>

      <p
        v-if="freeTextSubmission"
        class="guided-composer-submission"
        :class="`is-${freeTextSubmission.status}`"
        :role="freeTextSubmission.status === 'failed' ? 'alert' : 'status'"
        aria-live="polite"
      >
        <strong>你说</strong>
        <span class="guided-composer-submitted-text" :title="freeTextSubmission.text">
          “{{ freeTextSubmission.text }}”
        </span>
        <span class="guided-composer-status" :title="composerStatusText">
          {{ composerStatusText }}
        </span>
      </p>

      <div class="guided-composer-row">
        <input
          id="guided-conversation-input"
          ref="searchInputRef"
          v-model="composerText"
          type="text"
          autocomplete="off"
          placeholder="例如：路牌上是日文，车辆靠左行驶；也可直接输入地名"
          :disabled="busy"
          aria-describedby="guided-composer-hint"
          @compositionstart="composing = true"
          @compositionend="composing = false"
          @keydown.enter="onComposerEnter"
        />
        <button type="submit" class="primary-btn" :disabled="!canSendComposer">
          {{ applying ? '发送中…' : busy ? '处理中…' : '发送线索' }}
        </button>
      </div>

      <p id="guided-composer-hint" class="guided-composer-hint">{{ composerHint }}</p>
    </form>
  </section>
</template>

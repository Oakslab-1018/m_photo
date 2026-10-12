<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type {
  GuidedDialogue,
  GuidedEventReceipt,
  GuidedLocationLevel,
  GuidedPhase,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import { normalizeIsoCode } from '@/utils/geo'
import {
  GUIDED_STEPS,
  candidatePointLabel,
  candidateScoreLabel,
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
  selectTopGuidedCandidates,
  targetLabel,
} from '@/utils/guided'

const props = withDefaults(
  defineProps<{
    dialogue: GuidedDialogue | null
    candidates: LocationCandidate[]
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
    loading: false,
    applying: false,
    externalBusy: false,
    error: null,
    conflict: null,
    canRetry: false,
    receipt: null,
    selectedGuidedTargetKey: null,
    suggestedFreeText: null,
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

const searchText = ref('')
const backtrackPrompt = ref<GuidedLocationLevel | null>(null)
const composing = ref(false)
const panelRef = ref<HTMLElement | null>(null)
const questionSectionRef = ref<HTMLElement | null>(null)
const candidateSectionRef = ref<HTMLElement | null>(null)

const stage = computed(() => props.dialogue?.stage ?? 'country')
const busy = computed(() => props.loading || props.applying || props.externalBusy)
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
/** Prominent backend backtrack target for admin1/city/resolved. */
const previousLevel = computed(() => guidedPreviousLevel(props.dialogue?.stage))
const nextStepLabel = computed(() =>
  stage.value === 'resolved' ? '' : `下一步：查看${locationLevelLabel(stage.value)}候选`,
)

const progressText = computed(() => {
  if (isResolved.value) return '3 / 3 · 已完成'
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
  if (stage.value === 'admin1') return '省份 / 地区候选'
  if (stage.value === 'city') return '城市候选'
  return '候选地点'
})

const searchLabel = computed(() => {
  if (!props.candidatesUnlocked) return '补充可观察线索'
  if (stage.value === 'country') return '输入国家 / 地区名称进行目录检索'
  if (stage.value === 'admin1') return '输入省份 / 地区名称进行目录检索'
  return '输入城市名称进行目录检索'
})

const searchPlaceholder = computed(() => {
  if (!props.candidatesUnlocked) return '描述照片中可见的招牌、地名或文字'
  if (stage.value === 'country') return '例如：日本 / Japan'
  if (stage.value === 'admin1') {
    const countryIso = normalizeIsoCode(props.dialogue?.confirmed.country?.id)
    return countryIso === 'CN'
      ? '例如：浙江 / Zhejiang'
      : '例如：Tokyo（使用目录中的英文名或别名）'
  }
  return '例如：Hangzhou、Tokyo'
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

function submitFreeText(): void {
  if (busy.value || composing.value) return
  const text = searchText.value.trim()
  if (!text) return
  emit('free-text', text)
}

function onSearchEnter(event: KeyboardEvent): void {
  if (event.isComposing || composing.value) return
  event.preventDefault()
  submitFreeText()
}

function requestRetry(): void {
  if (busy.value) return
  emit('retry')
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
    searchText.value = props.suggestedFreeText?.trim() ?? ''
    backtrackPrompt.value = null
  },
)

watch(
  () => props.suggestedFreeText,
  (value, previous) => {
    const suggested = value?.trim() ?? ''
    const previousSuggested = previous?.trim() ?? ''
    if (suggested) searchText.value = suggested
    else if (previousSuggested && searchText.value === previousSuggested) searchText.value = ''
  },
  { immediate: true },
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
  <section ref="panelRef" class="guided-panel" aria-live="polite">
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
        {{ safeGuidedReceipt(receipt.summary) || '操作已应用' }}
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

      <section v-if="!isResolved" ref="questionSectionRef" class="guided-question">
        <template v-if="phase === 'review'">
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

      <section v-if="candidatesPhase" ref="candidateSectionRef" class="guided-candidates">
        <div class="guided-section-head">
          <span class="guided-section-title">{{ candidateTitle }}</span>
          <span class="guided-section-note">候选可信度前五</span>
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

        <p class="guided-section-hint">
          {{ locationConfirmationReady
            ? '候选可信度前五 · 排序分仅供参考，不代表概率'
            : '新增观察待核对，候选排名仍可查看，但暂不能确认或排除地点。' }}
        </p>
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

      <section v-if="!isResolved" class="guided-search">
        <label for="guided-catalog-search">{{ searchLabel }}</label>
        <div class="guided-search-row">
          <input
            id="guided-catalog-search"
            v-model="searchText"
            type="text"
            :placeholder="searchPlaceholder"
            :disabled="busy"
            @compositionstart="composing = true"
            @compositionend="composing = false"
            @keydown.enter="onSearchEnter"
          />
          <button type="button" class="primary-btn" :disabled="busy || !searchText.trim()" @click="submitFreeText">
            {{ applying ? '提交中…' : '搜索 / 提交文字' }}
          </button>
        </div>
        <p>自由文本只用于目录检索或提出线索，不会自动确认地点或进入下一级。</p>
      </section>
    </template>

    <div v-else-if="!loading" class="guided-loading">
      引导状态尚未就绪，请重试。
    </div>
  </section>
</template>
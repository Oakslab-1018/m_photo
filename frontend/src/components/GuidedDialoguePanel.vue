<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type {
  GuidedDialogue,
  GuidedEventReceipt,
  GuidedLocationLevel,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import { confidencePercent } from '@/utils/clues'
import { normalizeIsoCode } from '@/utils/geo'
import {
  GUIDED_STEPS,
  candidatePointLabel,
  candidateScoreLabel,
  guidedQuestionMessage,
  guidedQuestionReason,
  guidedQuestionTitle,
  guidedStageTitle,
  guidedStepState,
  guidedTargetKey,
  locationLevelLabel,
  pendingQuestionClue,
  reviewedClueStatusLabel,
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
    /** Server candidate rankings are hidden until the review round completes. */
    candidatesUnlocked?: boolean
    /** New observations may temporarily block only the location actions. */
    locationConfirmationReady?: boolean
    /** Newly appended clue ids are highlighted and lifted to the top. */
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
    candidatesUnlocked: true,
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
const pendingClue = computed(() =>
  props.dialogue ? pendingQuestionClue(props.dialogue) : null,
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
  if (!props.candidatesUnlocked && question.kind !== 'review_clue') return []
  return (question.options ?? []).filter(
    (option) => typeof option?.value === 'string' && typeof option?.label === 'string',
  )
})

// Location confirmation is always an explicit confirm_location event. The
// backend also labels question options with candidate IDs, but sending those as
// `answer` would only add a place-name clue, so the candidate cards below own
// confirmation. Clue review likewise uses the explicit clue actions.
const answerChoiceOptions = computed(() => {
  const kind = pendingQuestion.value?.kind
  if (kind !== 'ask' && kind !== 'request_clue') return []
  return questionOptions.value
})

const pendingClueCount = computed(
  () => props.dialogue?.clues.filter((clue) => clue.status === 'pending').length ?? 0,
)
const reviewRoundComplete = computed(
  () => props.candidatesUnlocked && pendingClueCount.value === 0 && !isResolved.value,
)
/** Defense in depth: even a caller passing the full server list cannot render past Top-5. */
const topCandidates = computed(() =>
  stage.value === 'resolved'
    ? []
    : selectTopGuidedCandidates(props.candidates, stage.value),
)

const orderedClues = computed(() => {
  const recent = new Set(props.recentClueIds)
  return [...(props.dialogue?.clues ?? [])].sort((left, right) => {
    const leftRank = recent.has(left.id) ? 0 : left.status === 'pending' ? 1 : 2
    const rightRank = recent.has(right.id) ? 0 : right.status === 'pending' ? 1 : 2
    return leftRank - rightRank
  })
})

function isRecentClue(clueId: string): boolean {
  return props.recentClueIds.includes(clueId)
}

const candidateTitle = computed(() => {
  if (stage.value === 'country') return '国家 / 地区候选'
  if (stage.value === 'admin1') return '省份 / 地区候选'
  if (stage.value === 'city') return '城市候选'
  return '候选地点'
})

const nextStageActionLabel = computed(() => {
  if (stage.value === 'country') return '进入国家 / 地区筛选'
  if (stage.value === 'admin1') return '进入省份 / 地区筛选'
  if (stage.value === 'city') return '进入城市筛选'
  return ''
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
  return guidedQuestionMessage(question, pendingClue.value, props.candidatesUnlocked)
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

function enterCurrentStageCandidates(): void {
  scrollToCandidates()
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

watch(
  [() => props.candidatesUnlocked, pendingClueCount],
  ([unlocked, count], previous) => {
    const wasUnlocked = previous?.[0] ?? false
    const previousCount = previous?.[1] ?? count
    if ((!wasUnlocked && unlocked) || (previousCount > 0 && count === 0)) {
      scrollToCandidates()
    }
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
      <span class="guided-progress">{{ progressText }}</span>
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
        <p>已由服务器按确认路径完成定位；城市坐标仅为代表点，不是照片 GPS。</p>
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

      <section v-if="pendingQuestion && !isResolved" ref="questionSectionRef" class="guided-question">
        <div class="guided-section-head">
          <span class="guided-section-title">{{ guidedQuestionTitle(pendingQuestion) }}</span>
          <span class="guided-one-question">本轮仅此一问</span>
        </div>
        <p class="guided-question-message">{{ questionMessage }}</p>
        <p class="guided-question-reason">{{ guidedQuestionReason(pendingQuestion) }}</p>

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

        <div v-if="pendingQuestion.kind === 'review_clue' && pendingClue" class="guided-question-clue">
          <strong>{{ reviewedClueText(pendingClue) }}</strong>
          <span>{{ pendingClue.description || '未提供详细说明' }}</span>
          <div class="guided-inline-actions">
            <button type="button" class="mini-btn confirm" :disabled="busy" @click="emit('confirm-clue', pendingClue.id)">
              判断准确
            </button>
            <button type="button" class="mini-btn reject" :disabled="busy" @click="emit('reject-clue', pendingClue.id)">
              不准确
            </button>
          </div>
        </div>

        <div
          v-else-if="!candidatesUnlocked"
          class="guided-question-hint"
        >
          先完成上方观察核对；核对前不会展示可能包含地点名称的问题或候选。
        </div>
        <div v-else-if="pendingQuestion.kind === 'confirm_location'" class="guided-question-hint">
          请在下方候选中明确确认当前层级；不会根据排序第一名自动确认。
        </div>
        <div v-else-if="!answerChoiceOptions.length" class="guided-question-hint">
          可在下方输入可观察线索；自由文本不会直接推进确认层级。
        </div>

        <button
          v-if="candidatesUnlocked || pendingQuestion.kind === 'review_clue'"
          type="button"
          class="guided-unknown-btn"
          :disabled="busy"
          @click="emit('unknown')"
        >
          不确定 / 无法判断（不代表排除）
        </button>
      </section>

      <section v-if="!isResolved" ref="candidateSectionRef" class="guided-candidates">
        <div class="guided-section-head">
          <span class="guided-section-title">{{ candidateTitle }}</span>
          <span class="guided-section-note">服务器返回的目标</span>
        </div>

        <div v-if="!candidatesUnlocked" class="guided-candidate-lock" role="status">
          <strong>先核对观察，再查看地点候选</strong>
          <span>还有 {{ pendingClueCount }} 条观察待处理；核对完成前不会展示国家 / 地区、省份或城市候选。</span>
        </div>

        <template v-else>
          <div v-if="reviewRoundComplete" class="guided-next-step" role="status">
            <div>
              <strong>观察核对已完成</strong>
              <span v-if="topCandidates.length">下一步请手动查看当前层级候选；系统不会自动确认地点。</span>
              <span v-else>当前层级暂无候选。请在下方补充照片中可见的招牌、地名或文字，系统会据此重新检索。</span>
            </div>
            <button
              v-if="topCandidates.length && nextStageActionLabel"
              type="button"
              class="primary-btn guided-next-step-action"
              @click="enterCurrentStageCandidates"
            >
              {{ nextStageActionLabel }}
            </button>
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
        </template>
      </section>

      <section v-if="dialogue.clues.length" class="guided-clues">
        <div class="guided-section-head">
          <span class="guided-section-title">证据核对</span>
          <span class="guided-section-note">观察与地点分开处理</span>
        </div>
        <p class="guided-section-hint">
          确认观察不等于确认地点；排除证据也不等于地理排除。新增线索会置顶并高亮显示。
        </p>
        <div class="guided-clue-list" aria-label="观察核对列表">
          <article
            v-for="clue in orderedClues"
            :key="clue.id"
            class="guided-clue-card"
            :class="[`status-${clue.status}`, { 'is-new': isRecentClue(clue.id) }]"
          >
            <div class="guided-clue-main">
              <strong>{{ reviewedClueText(clue) }}</strong>
              <span class="guided-clue-status">{{ reviewedClueStatusLabel(clue.status) }}</span>
              <span v-if="isRecentClue(clue.id)" class="guided-clue-new">新线索</span>
              <span v-if="typeof clue.confidence === 'number' && Number.isFinite(clue.confidence)" class="guided-clue-confidence">
                证据置信度 {{ confidencePercent(clue.confidence) }}
              </span>
            </div>
            <p v-if="clue.description" class="guided-clue-description">{{ clue.description }}</p>
            <div v-if="clue.status === 'pending'" class="guided-inline-actions">
              <button type="button" class="mini-btn confirm" :disabled="busy" @click="emit('confirm-clue', clue.id)">
                判断准确
              </button>
              <button type="button" class="mini-btn reject" :disabled="busy" @click="emit('reject-clue', clue.id)">
                不准确
              </button>
            </div>
            <button
              v-else-if="clue.status === 'rejected' || clue.status === 'uncertain'"
              type="button"
              class="guided-link-btn"
              :disabled="busy"
              @click="emit('retract-clue', clue.id)"
            >
              恢复待核对
            </button>
          </article>
        </div>
      </section>

      <section v-if="candidatesUnlocked && dialogue.exclusions.length" class="guided-exclusions">
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


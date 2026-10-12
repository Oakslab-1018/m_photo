<script setup lang="ts">
import { computed, ref, watch } from 'vue'
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
const showAllClues = ref(false)
const composing = ref(false)

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

const questionOptions = computed(() =>
  (pendingQuestion.value?.options ?? []).filter(
    (option) => typeof option?.value === 'string' && typeof option?.label === 'string',
  ),
)

// Location confirmation is always an explicit confirm_location event. The
// backend also labels question options with candidate IDs, but sending those as
// `answer` would only add a place-name clue, so the candidate cards below own
// confirmation. Clue review likewise uses the explicit clue actions.
const answerChoiceOptions = computed(() => {
  const kind = pendingQuestion.value?.kind
  if (kind !== 'ask' && kind !== 'request_clue') return []
  return questionOptions.value
})

const visibleClues = computed(() => {
  const clues = props.dialogue?.clues ?? []
  return showAllClues.value ? clues : clues.slice(0, 6)
})

const hiddenClueCount = computed(() => Math.max(0, (props.dialogue?.clues.length ?? 0) - visibleClues.value.length))

const candidateTitle = computed(() => {
  if (stage.value === 'country') return '国家 / 地区候选'
  if (stage.value === 'admin1') return '省份 / 地区候选'
  if (stage.value === 'city') return '城市候选'
  return '候选地点'
})

const searchLabel = computed(() => {
  if (stage.value === 'country') return '输入国家 / 地区名称进行目录检索'
  if (stage.value === 'admin1') return '输入省份 / 地区名称进行目录检索'
  return '输入城市名称进行目录检索'
})

const searchPlaceholder = computed(() => {
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
    showAllClues.value = false
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
</script>

<template>
  <section class="guided-panel" aria-live="polite">
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

      <section v-if="pendingQuestion && !isResolved" class="guided-question">
        <div class="guided-section-head">
          <span class="guided-section-title">{{ guidedQuestionTitle(pendingQuestion) }}</span>
          <span class="guided-one-question">本轮仅此一问</span>
        </div>
        <p class="guided-question-message">{{ pendingQuestion.message }}</p>
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
              确认这条观察
            </button>
            <button type="button" class="mini-btn reject" :disabled="busy" @click="emit('reject-clue', pendingClue.id)">
              这条证据不可靠
            </button>
          </div>
        </div>

        <div v-else-if="pendingQuestion.kind === 'confirm_location'" class="guided-question-hint">
          请在下方候选中明确确认当前层级；不会根据排序第一名自动确认。
        </div>
        <div v-else-if="!answerChoiceOptions.length" class="guided-question-hint">
          可在下方输入可观察线索；自由文本不会直接推进确认层级。
        </div>

        <button type="button" class="guided-unknown-btn" :disabled="busy" @click="emit('unknown')">
          不确定 / 无法判断（不代表排除）
        </button>
      </section>

      <section v-if="!isResolved" class="guided-candidates">
        <div class="guided-section-head">
          <span class="guided-section-title">{{ candidateTitle }}</span>
          <span class="guided-section-note">服务器返回的目标</span>
        </div>
        <p class="guided-section-hint">单击候选卡片或地图标记只在图上高亮定位；确认或排除只作用于后端返回的具体目标，排序分不是概率。</p>
        <div v-if="!candidates.length" class="guided-empty">
          暂无候选。可通过下方文字检索目录，搜索不会自动确认地点。
        </div>
        <article
          v-for="candidate in candidates"
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
              :disabled="busy"
              @click.stop="emit('confirm-location', candidate.target)"
            >
              确认{{ locationLevelLabel(candidate.target.level) }}
            </button>
            <button
              type="button"
              class="mini-btn reject"
              :disabled="busy"
              @click.stop="emit('reject-location', candidate.target)"
            >
              排除{{ locationLevelLabel(candidate.target.level) }}
            </button>
          </div>
        </article>
      </section>

      <section v-if="dialogue.clues.length" class="guided-clues">
        <div class="guided-section-head">
          <span class="guided-section-title">证据核对</span>
          <span class="guided-section-note">观察与地点分开处理</span>
        </div>
        <p class="guided-section-hint">
          确认观察不等于确认地点；排除证据也不等于地理排除。
        </p>
        <article v-for="clue in visibleClues" :key="clue.id" class="guided-clue-card" :class="`status-${clue.status}`">
          <div class="guided-clue-main">
            <strong>{{ reviewedClueText(clue) }}</strong>
            <span class="guided-clue-status">{{ reviewedClueStatusLabel(clue.status) }}</span>
            <span v-if="typeof clue.confidence === 'number' && Number.isFinite(clue.confidence)" class="guided-clue-confidence">
              证据置信度 {{ confidencePercent(clue.confidence) }}
            </span>
          </div>
          <p v-if="clue.description" class="guided-clue-description">{{ clue.description }}</p>
          <div v-if="clue.status === 'pending'" class="guided-inline-actions">
            <button type="button" class="mini-btn confirm" :disabled="busy" @click="emit('confirm-clue', clue.id)">
              确认这条观察
            </button>
            <button type="button" class="mini-btn reject" :disabled="busy" @click="emit('reject-clue', clue.id)">
              证据不可靠
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
        <button v-if="hiddenClueCount" type="button" class="guided-link-btn" @click="showAllClues = true">
          展开其余 {{ hiddenClueCount }} 条证据
        </button>
      </section>

      <section v-if="dialogue.exclusions.length" class="guided-exclusions">
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


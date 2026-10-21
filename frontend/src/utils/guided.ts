import type {
  ConfidenceRoute,
  GuidedDialogue,
  GuidedEventReceipt,
  GuidedLocationLevel,
  GuidedQuestion,
  GuidedStage,
  GuidedTargetRef,
  HypothesisAction,
  LocationCandidate,
  LocationTarget,
  ReviewedClue,
  ReviewedClueStatus,
} from '@/types'
import { displayCityName } from '@/utils/city'
import { clueValueLabel, dimensionLabel } from '@/utils/clues'
import { normalizeIsoCode } from '@/utils/geo'

export interface GuidedStep {
  level: GuidedLocationLevel
  label: string
}

export const GUIDED_STEPS: GuidedStep[] = [
  { level: 'country', label: '国家 / 地区' },
  { level: 'city', label: '城市' },
]

/** Guided cards intentionally expose only the strongest five location options. */
export const GUIDED_CANDIDATE_LIMIT = 5

const LEVEL_LABELS: Record<GuidedLocationLevel, string> = {
  country: '国家 / 地区',
  // Deprecated internal level: retained for payload typing, never rendered.
  admin1: '地点',
  city: '城市',
}

const STAGE_TITLES: Record<GuidedStage, string> = {
  country: '先确认国家 / 地区',
  city: '确认城市',
  resolved: '已确认城市',
}

const LOCALIZED_DIMENSION_PATTERN = Object.values({
  language: '文字语言',
  signage: '招牌文字',
  place_name: '地名证据',
  road: '道路词汇',
  driving_side: '通行方向',
  terrain: '地形词汇',
  waterbody: '水体词汇',
  vegetation: '植被类型',
  building: '建筑风格',
  vehicle: '车辆特征',
  climate: '气候特征',
  culture: '文化符号',
  infrastructure: '基础设施',
  other: '其他线索',
}).join('|')

const QUESTION_TITLES: Record<GuidedQuestion['kind'], string> = {
  review_clue: '核对一条观察',
  ask: '补充一项线索',
  confirm_location: '确认地点层级',
  request_clue: '补充可观察线索',
}

export function guidedStageTitle(stage: GuidedStage): string {
  return STAGE_TITLES[stage]
}

function levelText(level: string): string {
  const normalized = level.toLowerCase()
  // Old persisted receipts may still mention admin1. The public path no longer
  // has that step, so render it as the city step rather than exposing it.
  if (normalized === 'admin1') return '城市'
  if (normalized === 'city') return '城市'
  return '国家 / 地区'
}

/** Sanitize known backend enum/audit phrases without exposing raw contract tokens. */
export function safeGuidedText(text: string | null | undefined): string {
  return (text ?? '')
    .replace(
      /\b([a-z][a-z0-9_]*)\s*=\s*([a-z][a-z0-9_]*)\b/gi,
      (_match, dimension: string, value: string) =>
        `${dimensionLabel(dimension.toLowerCase())}=${clueValueLabel(value.toLowerCase())}`,
    )
    .replace(
      new RegExp(`(${LOCALIZED_DIMENSION_PATTERN})\\s*=\\s*([a-z][a-z0-9_]*)`, 'gi'),
      (_match, dimension: string, value: string) => `${dimension}=${clueValueLabel(value.toLowerCase())}`,
    )
    .replace(/\bgeneric\b/gi, '普通 / 不明显')
    .replace(/通用\s*\/\s*不明显/g, '普通 / 不明显')
    .replace(
      /(回退到|当前)\s*(admin1|country|city)\s*阶段/gi,
      (_match, prefix: string, level: string) => `${prefix}${levelText(level)}阶段`,
    )
}

/** Receipt summaries may be short backend phrases; remove remaining internal tokens. */
export function safeGuidedReceipt(text: string | null | undefined): string {
  const value = safeGuidedText(text).trim()
  if (/^(admin1|country|city)$/i.test(value)) return levelText(value)
  return value.replace(/\badmin1\b/gi, '地点').replace(/\brevision\b/gi, '版本')
}

export function locationLevelLabel(level: GuidedLocationLevel): string {
  return LEVEL_LABELS[level]
}

export type GuidedFreeTextReceiptStatus = 'applied' | 'unchanged' | 'unconfirmed' | 'failed'

export interface GuidedFreeTextReceiptOutcome {
  status: GuidedFreeTextReceiptStatus
  error?: string | null
}

/**
 * Map a free-text receipt without conflating an already-saved clue with an
 * applied mutation or a failed submission. `unchanged` is an acknowledged
 * pre-existing clue, not an application failure and not a fabricated success.
 */
export function guidedFreeTextReceiptOutcome(
  receipt: GuidedEventReceipt | null | undefined,
): GuidedFreeTextReceiptOutcome {
  const status = receipt?.status?.trim().toLowerCase() ?? ''
  if (status === 'unrecognized') {
    return {
      status: 'failed',
      error: receipt?.summary ?? '没有识别到新的线索，请修改后重试。',
    }
  }
  if (status === 'unchanged') return { status: 'unchanged' }
  if (status === 'applied') return { status: 'applied' }
  return {
    status: 'unconfirmed',
    error: '服务器未返回可识别的回执，请核对最新状态后重试。',
  }
}

/** Stable key shared by candidate cards, App selection state, and map markers. */
export function guidedTargetKey(target: GuidedTargetRef | null | undefined): string | null {
  if (!target) return null
  const id = typeof target.id === 'string' ? target.id.trim() : ''
  if (!id) return null
  return target.level + ':' + id
}

export function sameGuidedTarget(
  left: GuidedTargetRef | null | undefined,
  right: GuidedTargetRef | null | undefined,
): boolean {
  const leftKey = guidedTargetKey(left)
  return leftKey !== null && leftKey === guidedTargetKey(right)
}

export function guidedStepState(
  stage: GuidedStage,
  level: GuidedLocationLevel,
): 'done' | 'current' | 'pending' {
  if (stage === 'resolved') return 'done'
  const stageIndex = GUIDED_STEPS.findIndex((step) => step.level === stage)
  const levelIndex = GUIDED_STEPS.findIndex((step) => step.level === level)
  if (levelIndex < stageIndex) return 'done'
  if (levelIndex === stageIndex) return 'current'
  return 'pending'
}

/** Human-readable target label; city titles follow the all-Chinese policy. */
export function targetLabel(target: LocationTarget | null | undefined): string {
  if (target?.level === 'city') return displayCityName(target)
  const label = typeof target?.label === 'string' ? target.label.trim() : ''
  return label || '未命名地点'
}

export function isCorrectionQuestion(question: GuidedQuestion): boolean {
  if (question.kind !== 'ask') return false
  const options = question.options ?? []
  const values = options.map((option) => option.value.trim().toLowerCase())
  const labels = options.map((option) => option.label.trim())
  const hasApply =
    values.includes('apply') ||
    values.includes('apply_correction') ||
    labels.some((label) => /应用|采纳|确认更新|更新证据/.test(label))
  const hasKeep =
    values.includes('keep') ||
    values.includes('keep_evidence') ||
    labels.some((label) => /保留|保持|不更新|维持/.test(label))
  const text = `${question.id} ${question.message}`
  return (hasApply && hasKeep) || /纠正|修正|否定|correction|negative/i.test(text)
}

export function guidedQuestionTitle(question: GuidedQuestion): string {
  if (isCorrectionQuestion(question)) return '核对纠正'
  return QUESTION_TITLES[question.kind]
}

/** Brief user-facing rationale for the one pending question. */
export function guidedQuestionReason(question: GuidedQuestion): string {
  if (isCorrectionQuestion(question)) {
    return '确认后才更新证据或排除记录。'
  }
  switch (question.kind) {
    case 'review_clue':
      return '请帮我们判断这条观察是否准确；确认观察不等于确认地点。'
    case 'ask':
      return '这项信息有助于区分当前候选，请选择最符合照片的答案。'
    case 'confirm_location':
      return '需要你明确确认当前层级；自由文本和排序结果都不会自动前进。'
    case 'request_clue':
      return '补充一项可观察线索有助于检索目录并缩小候选范围。'
  }
}

export function reviewedClueStatusLabel(status: ReviewedClueStatus): string {
  switch (status) {
    case 'pending':
      return '待核对'
    case 'confirmed':
      return '观察准确'
    case 'rejected':
      return '已标记不准确'
    case 'uncertain':
      return '暂不确定（未排除）'
  }
}

export function reviewedClueText(clue: ReviewedClue): string {
  return `${dimensionLabel(clue.dimension)} · ${clueValueLabel(clue.value)}`
}

/** All pending observations must be handled before a location can be confirmed. */
export function hasPendingReviewedClues(dialogue: GuidedDialogue | null | undefined): boolean {
  return Boolean(dialogue?.clues.some((clue) => clue.status === 'pending'))
}

/**
 * Blocking questions (ask/correction/request_clue) must be answered in the
 * review box before any location candidate may be shown. A confirm_location
 * question is not a review blocker: it is answered later by an explicit
 * candidate confirm, never by an `answer` chip.
 */
export function isBlockingGuidedQuestion(question: GuidedQuestion | null | undefined): boolean {
  return Boolean(question && question.kind !== 'confirm_location')
}

/** Country-only product route authorized by both the action kind and confidence action. */
export type CountryConfidenceShortcut = 'high' | 'candidates'

/**
 * The confidence route may bypass routine initial machine review, but only at
 * country stage and only when the authoritative action matches. Corrections,
 * conflicts and explicit user re-review are always supplied as blocked.
 */
export function countryConfidenceShortcut(
  dialogue: GuidedDialogue | null | undefined,
  action: HypothesisAction | null | undefined,
  confidence: ConfidenceRoute | null | undefined,
  blocked = false,
): CountryConfidenceShortcut | null {
  if (blocked || !dialogue || dialogue.stage !== 'country' || !action || !confidence) return null
  const questionKind = dialogue.pending_question?.kind
  if (questionKind === 'ask' || questionKind === 'request_clue') return null
  if (
    action.kind === 'result' &&
    confidence.tier === 'high' &&
    confidence.action === 'direct_top1'
  ) {
    return 'high'
  }
  if (
    action.kind === 'confirm' &&
    confidence.action === 'confirm_candidates' &&
    (confidence.tier === 'medium' || confidence.tier === 'low')
  ) {
    return 'candidates'
  }
  return null
}

/**
 * Join sparse confidence top3 rows back to server-provided catalog targets.
 * Never synthesize a target; missing catalog rows are omitted.
 */
export function confidenceCountryCandidates(
  confidence: ConfidenceRoute | null | undefined,
  candidates: readonly LocationCandidate[],
  limit: 1 | 3,
): LocationCandidate[] {
  if (!confidence) return []
  const seen = new Set<string>()
  const result: LocationCandidate[] = []
  for (const summary of confidence.top3.slice(0, limit)) {
    const code = normalizeIsoCode(summary.iso)
    if (!code || seen.has(code)) continue
    const candidate = candidates.find((item) => {
      if (item.target.level !== 'country') return false
      return [item.target.id, item.target.country_iso, item.target.source_iso].some(
        (value) => normalizeIsoCode(value ?? '') === code,
      )
    })
    if (!candidate) continue
    seen.add(code)
    result.push({
      ...candidate,
      score:
        typeof summary.score === 'number' && Number.isFinite(summary.score)
          ? summary.score
          : candidate.score,
      score_type: summary.score_type ?? candidate.score_type ?? null,
    })
  }
  return result
}

/** Every observation handled and no unresolved ask/correction left. */
export function guidedReviewReady(dialogue: GuidedDialogue | null | undefined): boolean {
  if (!dialogue || dialogue.stage === 'resolved') return false
  if (hasPendingReviewedClues(dialogue)) return false
  return !isBlockingGuidedQuestion(dialogue.pending_question)
}

/** The single observation the server currently pins for review, if any. */
export function guidedCurrentReviewClue(
  dialogue: GuidedDialogue | null | undefined,
): ReviewedClue | null {
  const question = dialogue?.pending_question
  if (!dialogue || !question || question.kind !== 'review_clue' || !question.clue_id) return null
  return dialogue.clues.find((clue) => clue.id === question.clue_id) ?? null
}

/** Position of the server-pinned review clue inside the snapshot, e.g. 第2/8. */
export function guidedReviewProgress(
  dialogue: GuidedDialogue | null | undefined,
): { current: number; total: number } | null {
  const clue = guidedCurrentReviewClue(dialogue)
  if (!clue || !dialogue) return null
  const index = dialogue.clues.findIndex((item) => item.id === clue.id)
  if (index < 0) return null
  return { current: index + 1, total: dialogue.clues.length }
}

/** The level a prominent 返回上一步 button backtracks to, or null at country. */
export function guidedPreviousLevel(
  stage: GuidedStage | null | undefined,
): GuidedLocationLevel | null {
  switch (stage) {
    case 'city':
      return 'country'
    case 'resolved':
      return 'city'
    default:
      return null
  }
}

function isUserReviewedClue(clue: ReviewedClue): boolean {
  // Snapshots from some deployments omit `source`; a missing provenance must
  // never throw (it only means the row cannot be trusted as user evidence).
  const source = (clue.source ?? '').trim().toLowerCase()
  return source === 'user' || source.startsWith('guided_user')
}

/**
 * Bounded rebuild of the sticky ranking unlock after a reload.
 *
 * All-pending snapshots stay locked. Pending machine evidence also stays
 * locked because a refresh cannot distinguish initial model observations from
 * newly appended machine evidence. If every pending row is user-originated and
 * the snapshot already contains a handled observation or confirmed path, the
 * initial machine review must have completed; rankings may reopen while the
 * user rows still block location confirmation.
 */
export function canUnlockReviewedRankings(dialogue: GuidedDialogue | null | undefined): boolean {
  if (!dialogue) return false
  const pending = dialogue.clues.filter((clue) => clue.status === 'pending')
  if (!pending.length) return true
  const pendingOnlyUser = pending.every(isUserReviewedClue)
  const hasHandledObservation = dialogue.clues.some((clue) => clue.status !== 'pending')
  // Legacy admin1 metadata is deliberately ignored: only the public
  // country -> city path can prove that a review round was completed.
  const hasConfirmedPath = Boolean(dialogue.confirmed.country || dialogue.confirmed.city)
  return pendingOnlyUser && (hasHandledObservation || hasConfirmedPath)
}

/** Natural Chinese prompt built from the server clue, never the raw enum message. */
export function guidedQuestionMessage(
  question: GuidedQuestion,
  clue?: ReviewedClue | null,
  locationCandidatesVisible = true,
): string {
  // Only the location-confirmation prompt may carry a place name that must stay
  // hidden until the candidate phase; blocking ask/correction prompts stay
  // readable in the review box so they can never be swallowed by the gate.
  if (!locationCandidatesVisible && question.kind === 'confirm_location') {
    return '下一步将展示地点候选，选择合适的地点后再确认。'
  }
  if (question.kind !== 'review_clue' || !clue) return safeGuidedText(question.message)
  const dimension = clue.dimension || question.dimension || ''
  const value = clueValueLabel(clue.value)
  if (dimension === 'building') {
    return `我们初步判断这是‘${value}’的建筑风格，您觉得准确吗？`
  }
  if (dimension === 'language') {
    return `我们初步判断画面中的文字语言是‘${value}’，您觉得准确吗？`
  }
  if (dimension === 'signage' || dimension === 'place_name') {
    return `我们初步判断画面中出现了‘${value}’，这项观察准确吗？`
  }
  if (dimension === 'driving_side') {
    return `我们初步判断这里可能是‘${value}’，您觉得准确吗？`
  }
  return `我们初步判断这是‘${value}’的${dimensionLabel(dimension)}，您觉得准确吗？`
}

/** Explicit server provenance outranks every model score. */
function selectionBasisRank(candidate: LocationCandidate): 0 | 1 | 2 {
  if (candidate.selection_basis === 'user_text') return 0
  if (candidate.selection_basis === 'catalog_search') return 1
  return 2
}

interface RankedGuidedCandidate {
  candidate: LocationCandidate
  index: number
  score: number | null
  basisRank: 0 | 1 | 2
}

function mergeGuidedCandidate(
  left: RankedGuidedCandidate,
  right: RankedGuidedCandidate,
): RankedGuidedCandidate {
  const leftScore = left.score
  const rightScore = right.score
  const scored =
    rightScore !== null && (leftScore === null || rightScore > leftScore) ? right : left
  const basis = right.basisRank < left.basisRank ? right : left
  const merged: LocationCandidate = {
    ...scored.candidate,
    ...(basis.candidate.selection_basis
      ? { selection_basis: basis.candidate.selection_basis }
      : {}),
  }
  const placeMatch = basis.candidate.place_match ?? left.candidate.place_match ?? right.candidate.place_match
  if (placeMatch) merged.place_match = placeMatch
  if (scored.score !== null) {
    merged.score = scored.score
    merged.score_type = scored.candidate.score_type ?? null
  } else {
    delete merged.score
    delete merged.score_type
  }
  const reason = left.candidate.reason ?? right.candidate.reason
  if (reason) merged.reason = reason
  return {
    candidate: merged,
    index: Math.min(left.index, right.index),
    score: scored.score,
    basisRank: basis.basisRank,
  }
}

/**
 * Deduplicate and rank guided candidates without inventing confidence. At the
 * country stage, authorized user/catalog matches retain their explicit
 * priority. At the city stage, every real finite score sorts together in
 * descending order; selection basis only breaks score ties or orders unscored
 * catalog rows. Legacy admin1 candidates never enter a public stage.
 */
export function selectTopGuidedCandidates(
  candidates: readonly LocationCandidate[],
  stage?: GuidedStage | null,
): LocationCandidate[] {
  const uniqueByKey = new Map<string, RankedGuidedCandidate>()
  for (const [index, candidate] of candidates.entries()) {
    if (!candidate || (stage && candidate.target.level !== stage)) continue
    const key = guidedTargetKey(candidate.target)
    if (!key) continue
    const score =
      typeof candidate.score === 'number' && Number.isFinite(candidate.score)
        ? candidate.score
        : null
    const entry: RankedGuidedCandidate = {
      candidate,
      index,
      score,
      basisRank: selectionBasisRank(candidate),
    }
    const current = uniqueByKey.get(key)
    uniqueByKey.set(key, current ? mergeGuidedCandidate(current, entry) : entry)
  }

  const unique = [...uniqueByKey.values()]
  if (stage === 'city') {
    const scored = unique
      .filter((entry) => entry.score !== null)
      .sort(
        (left, right) =>
          (right.score as number) - (left.score as number) ||
          left.basisRank - right.basisRank ||
          left.index - right.index,
      )
    const unscored = unique
      .filter((entry) => entry.score === null)
      .sort(
        (left, right) =>
          left.basisRank - right.basisRank || left.index - right.index,
      )
    return [...scored, ...unscored]
      .slice(0, GUIDED_CANDIDATE_LIMIT)
      .map(({ candidate }) => candidate)
  }

  const explicit = unique
    .filter((entry) => entry.basisRank < 2)
    .sort(
      (left, right) =>
        left.basisRank - right.basisRank ||
        (right.score ?? Number.NEGATIVE_INFINITY) - (left.score ?? Number.NEGATIVE_INFINITY) ||
        left.index - right.index,
    )
  const scored = unique
    .filter((entry) => entry.basisRank === 2 && entry.score !== null)
    .sort(
      (left, right) =>
        (right.score as number) - (left.score as number) || left.index - right.index,
    )
  const catalogOnly = unique.filter((entry) => entry.basisRank === 2 && entry.score === null)

  return [...explicit, ...scored, ...catalogOnly]
    .slice(0, GUIDED_CANDIDATE_LIMIT)
    .map(({ candidate }) => candidate)
}

export interface GuidedCandidateComposition {
  hasUserText: boolean
  hasCatalogSearch: boolean
  hasExplicitSelectionBasis: boolean
  hasModelScores: boolean
  mixed: boolean
}

/** Shared ranking-label input for candidate panels and the guided dialogue. */
export function guidedCandidateComposition(
  candidates: readonly LocationCandidate[],
): GuidedCandidateComposition {
  const hasUserText = candidates.some((candidate) => candidate.selection_basis === 'user_text')
  const hasCatalogSearch = candidates.some(
    (candidate) => candidate.selection_basis === 'catalog_search',
  )
  const hasExplicitSelectionBasis = hasUserText || hasCatalogSearch
  const hasModelScores = candidates.some(
    (candidate) => typeof candidate.score === 'number' && Number.isFinite(candidate.score),
  )
  return {
    hasUserText,
    hasCatalogSearch,
    hasExplicitSelectionBasis,
    hasModelScores,
    mixed: hasExplicitSelectionBasis && hasModelScores,
  }
}

function scoreTypeLabel(scoreType: string | null | undefined): string {
  switch ((scoreType ?? '').trim()) {
    case 'city_text_cosine':
      return '城市文本余弦排序'
    case 'city_reference_hybrid':
      return '参考图融合排序'
    case 'city_clue_fusion':
      return '线索融合排序'
    default:
      return '匹配排序'
  }
}

/** Never turn missing provenance or a missing score into a fabricated probability. */
export function candidateScoreLabel(candidate: LocationCandidate): string {
  const score = candidate.score
  const scoreText =
    typeof score === 'number' && Number.isFinite(score)
      ? `${scoreTypeLabel(candidate.score_type)} ${score.toFixed(3)}（排序分，非概率）`
      : ''
  if (candidate.selection_basis === 'user_text') {
    return scoreText
      ? `线索匹配候选 · 来自你提供的文字线索；${scoreText}`
      : '线索匹配候选 · 来自你提供的文字线索（非模型概率）'
  }
  if (candidate.selection_basis === 'catalog_search') {
    return scoreText
      ? `目录检索匹配 · 来自目录名称查询；${scoreText}`
      : '目录检索匹配 · 来自目录名称查询（非模型概率）'
  }
  return scoreText || '目录候选（无模型评分）'
}

export function candidatePointLabel(target: LocationTarget): string {
  const lat = target.lat
  const lng = target.lng
  const hasPoint =
    typeof lat === 'number' &&
    Number.isFinite(lat) &&
    lat >= -90 &&
    lat <= 90 &&
    typeof lng === 'number' &&
    Number.isFinite(lng) &&
    lng >= -180 &&
    lng <= 180 &&
    !(Math.abs(lat) < 1e-9 && Math.abs(lng) < 1e-9)
  if (!hasPoint) return ''
  return '坐标为代表点，非照片 GPS'
}

export function pendingQuestionClue(dialogue: GuidedDialogue): ReviewedClue | null {
  const clueId = dialogue.pending_question?.clue_id
  if (!clueId) return null
  return dialogue.clues.find((clue) => clue.id === clueId) ?? null
}


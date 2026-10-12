import type {
  GuidedDialogue,
  GuidedLocationLevel,
  GuidedQuestion,
  GuidedStage,
  GuidedTargetRef,
  LocationCandidate,
  LocationTarget,
  ReviewedClue,
  ReviewedClueStatus,
} from '@/types'
import { clueValueLabel, dimensionLabel } from '@/utils/clues'

export interface GuidedStep {
  level: GuidedLocationLevel
  label: string
}

export const GUIDED_STEPS: GuidedStep[] = [
  { level: 'country', label: '国家 / 地区' },
  { level: 'admin1', label: '省份 / 地区' },
  { level: 'city', label: '城市' },
]

const LEVEL_LABELS: Record<GuidedLocationLevel, string> = {
  country: '国家 / 地区',
  admin1: '省份 / 地区',
  city: '城市',
}

const STAGE_TITLES: Record<GuidedStage, string> = {
  country: '先确认国家 / 地区',
  admin1: '确认省份 / 地区',
  city: '确认城市',
  resolved: '已确认城市',
}

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
  if (normalized === 'admin1') return '省份 / 地区'
  if (normalized === 'city') return '城市'
  return '国家 / 地区'
}

/** Sanitize known backend audit phrases without rewriting user free text. */
export function safeGuidedText(text: string | null | undefined): string {
  return (text ?? '').replace(
    /(回退到|当前)\s*(admin1|country|city)\s*阶段/gi,
    (_match, prefix: string, level: string) => `${prefix}${levelText(level)}阶段`,
  )
}

/** Receipt summaries may be short backend phrases; remove remaining internal tokens. */
export function safeGuidedReceipt(text: string | null | undefined): string {
  const value = safeGuidedText(text).trim()
  if (/^(admin1|country|city)$/i.test(value)) return levelText(value)
  // admin1 is never a legitimate user-facing location label in this catalog.
  return value.replace(/\badmin1\b/gi, '省份 / 地区').replace(/\brevision\b/gi, '版本')
}

export function locationLevelLabel(level: GuidedLocationLevel): string {
  return LEVEL_LABELS[level]
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

/** Human-readable target label; never falls back to an internal catalog ID. */
export function targetLabel(target: LocationTarget | null | undefined): string {
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
      return '这条观察会影响后续候选排序；确认观察不等于确认地点。'
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
      return '已确认证据'
    case 'rejected':
      return '已排除证据'
    case 'uncertain':
      return '不确定（不排除）'
  }
}

export function reviewedClueText(clue: ReviewedClue): string {
  return `${dimensionLabel(clue.dimension)} · ${clueValueLabel(clue.value)}`
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

/** Never turn a missing score into a fabricated probability. */
export function candidateScoreLabel(candidate: LocationCandidate): string {
  const score = candidate.score
  if (typeof score === 'number' && Number.isFinite(score)) {
    return `${scoreTypeLabel(candidate.score_type)} ${score.toFixed(3)}（排序分，非概率）`
  }
  return '目录候选（无模型评分）'
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
  if (target.level === 'admin1') {
    const derived =
      target.coordinate_source === 'admin1-largest-city-v1' ||
      target.coordinate_quality === 'derived'
    return derived
      ? '省级代表点由目录最大城市坐标派生，非行政区多边形，非照片 GPS'
      : '省级目录代表点，非行政区多边形，非照片 GPS'
  }
  return '坐标为代表点，非照片 GPS'
}

export function pendingQuestionClue(dialogue: GuidedDialogue): ReviewedClue | null {
  const clueId = dialogue.pending_question?.clue_id
  if (!clueId) return null
  return dialogue.clues.find((clue) => clue.id === clueId) ?? null
}


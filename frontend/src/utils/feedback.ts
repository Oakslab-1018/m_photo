import type { ChatMessage, HypothesisAction } from '@/types'
import { exclusionLabel, normalizeCountryIso } from '@/utils/geo'

/**
 * Feedback receipt rendering (`action.payload.feedback_receipt` and AI
 * `meta.feedback_receipt`). The backend contract is
 * `status: applied | unchanged | unrecognized` + `summary`, plus optional
 * added/removed clue/exclusion/leader diagnostics. Every extra field is
 * optional and shape-tolerant; a missing receipt never renders as success.
 */
export type FeedbackReceiptTone = 'ok' | 'muted' | 'warn'

export interface FeedbackReceiptView {
  status: string
  label: string
  tone: FeedbackReceiptTone
  summary: string | null
  details: string[]
}

type JsonRecord = Record<string, unknown>

const STATUS_VIEW: Record<string, { label: string; tone: FeedbackReceiptTone }> = {
  applied: { label: '已理解并应用', tone: 'ok' },
  unchanged: { label: '已理解，结论无变化', tone: 'muted' },
  unrecognized: { label: '未能识别这条反馈', tone: 'warn' },
  // Tolerated aliases so an older or adjacent backend never renders raw codes.
  understood: { label: '已理解', tone: 'ok' },
  accepted: { label: '已接收并理解', tone: 'ok' },
  no_change: { label: '已理解，结论无变化', tone: 'muted' },
  unknown: { label: '未能识别这条反馈', tone: 'warn' },
  failed: { label: '反馈处理失败', tone: 'warn' },
  error: { label: '反馈处理失败', tone: 'warn' },
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function firstOf(record: JsonRecord, keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key]
  }
  return undefined
}

function countOf(value: unknown): number | null {
  if (Array.isArray(value)) return value.length
  if (typeof value === 'number' && Number.isFinite(value)) return value
  const record = isRecord(value) ? value : null
  if (record) {
    const count = firstOf(record, ['count', 'total', 'n'])
    return typeof count === 'number' && Number.isFinite(count) ? count : null
  }
  return null
}

/** Raw exclusion codes keep their regional meaning; no TW -> CN folding here. */
function exclusionList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  for (const item of value) {
    const raw = typeof item === 'string'
      ? item
      : asString(firstOf(isRecord(item) ? item : {}, ['iso', 'source_iso']))
    if (!raw) continue
    const label = exclusionLabel(raw)
    if (label && !out.includes(label)) out.push(label)
  }
  return out
}

function leaderCode(value: unknown): string | null {
  const direct = asString(value)
  if (direct) return normalizeCountryIso(direct) || null
  const record = isRecord(value) ? value : null
  const iso = record ? asString(firstOf(record, ['iso', 'source_iso'])) : null
  return iso ? normalizeCountryIso(iso) || null : null
}

function receiptDetails(record: JsonRecord): string[] {
  const details: string[] = []
  const addedClues = countOf(firstOf(record, ['added_clues', 'addedClues']))
  const removedClues = countOf(firstOf(record, ['removed_clues', 'removedClues']))
  if (addedClues) details.push(`新增线索 ${addedClues} 条`)
  if (removedClues) details.push(`移除线索 ${removedClues} 条`)

  const addedExclusions = exclusionList(
    firstOf(record, ['added_exclusions', 'addedExclusions', 'new_exclusions', 'exclusions_added']),
  )
  const removedExclusions = exclusionList(
    firstOf(record, ['removed_exclusions', 'removedExclusions', 'cleared_exclusions', 'exclusions_removed']),
  )
  if (addedExclusions.length) details.push(`新增排除：${addedExclusions.join('、')}`)
  if (removedExclusions.length) details.push(`恢复候选：${removedExclusions.join('、')}`)

  const ranking: JsonRecord | null = isRecord(record.ranking_change)
    ? record.ranking_change
    : isRecord(record.rankingChange)
      ? record.rankingChange
      : null
  const leaderBefore =
    leaderCode(firstOf(record, ['leader_before', 'previous_leader', 'leader_from'])) ??
    (ranking ? leaderCode(firstOf(ranking, ['before', 'leader_before', 'from'])) : null)
  const leaderAfter =
    leaderCode(firstOf(record, ['leader_after', 'leader_iso', 'leader_to', 'leader'])) ??
    (ranking ? leaderCode(firstOf(ranking, ['after', 'leader_after', 'to', 'leader'])) : null)
  const leaderChanged =
    record.leader_changed === true ||
    record.leaderChanged === true ||
    ranking?.changed === true
  if (leaderBefore && leaderAfter && leaderBefore !== leaderAfter) {
    details.push(`领先候选：${leaderBefore} → ${leaderAfter}`)
  } else if (leaderAfter) {
    details.push(`领先候选：${leaderAfter}`)
  } else if (leaderChanged) {
    details.push('领先候选已更新')
  }
  return details
}

export function readFeedbackReceipt(raw: unknown): FeedbackReceiptView | null {
  const record = isRecord(raw) ? raw : null
  if (!record) {
    const text = asString(raw)
    return text ? { status: '', label: '服务器回执', tone: 'muted', summary: text, details: [] } : null
  }
  const status = (asString(record.status) ?? '').toLowerCase()
  const summary = asString(record.summary) ?? asString(record.detail) ?? asString(record.message)
  if (!status && !summary) return null
  const view = STATUS_VIEW[status]
  const questionAnswered =
    record.question === true || record.question_answered === true || record.questionAnswered === true
  let label = view?.label ?? (status ? `回执：${status}` : '服务器回执')
  let tone: FeedbackReceiptTone = view?.tone ?? 'muted'
  // Answering the pending question without new clues is a normal outcome, not
  // an unrecognized feedback: keep it muted and explain why.
  if (questionAnswered && (status === 'unrecognized' || status === 'unknown')) {
    label = '已回应问题（未新增线索）'
    tone = 'muted'
  }
  return {
    status,
    label,
    tone,
    summary: summary ?? null,
    details: receiptDetails(record),
  }
}

export function receiptFromMessage(message: ChatMessage | null | undefined): FeedbackReceiptView | null {
  const meta = isRecord(message?.meta) ? message.meta : null
  return meta ? readFeedbackReceipt(meta.feedback_receipt) : null
}

export function receiptFromAction(action: HypothesisAction | null | undefined): FeedbackReceiptView | null {
  return readFeedbackReceipt(action?.payload?.feedback_receipt)
}

/** Newest response receipt: the returned message tails first, then the action payload. */
export function feedbackReceipt(
  messages: readonly ChatMessage[] | null | undefined,
  action: HypothesisAction | null | undefined,
): FeedbackReceiptView | null {
  const list = messages ?? []
  for (let index = list.length - 1; index >= 0; index -= 1) {
    const receipt = receiptFromMessage(list[index])
    if (receipt) return receipt
  }
  return receiptFromAction(action)
}

export function receiptText(receipt: FeedbackReceiptView): string {
  const parts: string[] = [receipt.label]
  if (receipt.summary) parts.push(receipt.summary)
  if (receipt.details.length) parts.push(receipt.details.join(' · '))
  return parts.filter(Boolean).join(' · ')
}
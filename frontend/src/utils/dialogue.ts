import type { FeedbackPayload, HypothesisAction } from '@/types'
import { clueValueLabel, dimensionLabel } from '@/utils/clues'

/** Normalized answer chip shown under a pending dialogue question. */
export interface DialogueOption {
  dimension: string
  value: string
  label: string
}

export interface DialogueOptionGroup {
  dimension: string
  label: string
  options: DialogueOption[]
}

/**
 * Read structured answer options from an action payload. Malformed or empty
 * entries are dropped instead of being rendered; labels fall back to the
 * localized clue value so unknown dimensions never break the chips.
 */
export function actionOptions(action: HypothesisAction | null | undefined): DialogueOption[] {
  const raw = action?.payload?.options
  if (!Array.isArray(raw)) return []
  const seen = new Set<string>()
  const options: DialogueOption[] = []
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const dimension = typeof entry.dimension === 'string' ? entry.dimension.trim() : ''
    const value = typeof entry.value === 'string' ? entry.value.trim() : ''
    if (!dimension || !value) continue
    const key = `${dimension}::${value}`
    if (seen.has(key)) continue
    seen.add(key)
    const label = typeof entry.label === 'string' && entry.label.trim()
      ? entry.label.trim()
      : clueValueLabel(value)
    options.push({ dimension, value, label })
  }
  return options
}

/** Question-like actions get prompt visibility (auto-open chat / pending badge). */
export function isQuestionAction(action: HypothesisAction | null | undefined): boolean {
  if (!action) return false
  if (action.kind === 'ask' || action.kind === 'request_clue') return true
  return actionOptions(action).length > 0
}

/** Group normalized chips by dimension, preserving payload order. */
export function groupActionOptions(action: HypothesisAction | null | undefined): DialogueOptionGroup[] {
  const groups = new Map<string, DialogueOption[]>()
  for (const option of actionOptions(action)) {
    const items = groups.get(option.dimension) ?? []
    items.push(option)
    groups.set(option.dimension, items)
  }
  return [...groups.entries()].map(([dimension, options]) => ({
    dimension,
    label: dimensionLabel(dimension),
    options,
  }))
}

/** Feedback payload emitted when the user taps an answer chip. */
export function answerPayload(option: DialogueOption): FeedbackPayload {
  return { type: 'answer', dimension: option.dimension, value: option.value }
}
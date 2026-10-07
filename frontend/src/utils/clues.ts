import type { Clue } from '@/types'

export const DIMENSION_LABELS: Record<string, string> = {
  language: '文字语言',
  signage: '招牌文字',
  vegetation: '植被类型',
  building: '建筑风格',
  vehicle: '车辆特征',
  climate: '气候特征',
  culture: '文化符号',
  other: '其他线索',
}

export const VALUE_LABELS: Record<string, string> = {
  temperate_forest: '温带森林',
  grassland: '草原',
  savanna: '稀树草原',
  desert: '荒漠',
  tropical_rainforest: '热带雨林',
  urban_sparse: '城市稀疏植被',
  dark_vehicle: '深色车辆',
  white_silver_vehicle: '白/银色车辆',
  unknown_vehicle: '未识别车辆',
  mediterranean: '地中海风格',
  colonial: '殖民风格',
  gothic: '哥特式',
  modern_glass: '现代玻璃幕墙',
  generic: '通用/不明显',
  mediterranean_sunny: '地中海晴朗气候',
  temperate_maritime: '温带海洋性气候',
  cold_polar: '寒冷气候',
  arid: '干旱气候',
  temperate: '温带气候',
  zh: '中文',
  ja: '日语',
  ko: '韩语',
  latin: '拉丁语系',
  cyrillic: '西里尔语系',
  arabic: '阿拉伯语',
  thai: '泰语',
  unknown: '未知',
}

const DIMENSION_ORDER = [
  'language',
  'signage',
  'vegetation',
  'building',
  'vehicle',
  'climate',
  'culture',
  'other',
]

export interface ClueGroup {
  dimension: string
  label: string
  items: Clue[]
}

export function clueValueLabel(value: string): string {
  return VALUE_LABELS[value] ?? value
}

export function confidencePercent(confidence: number): string {
  const safe = Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0
  return `${Math.round(safe * 100)}%`
}

export function groupClues(clues: readonly Clue[]): ClueGroup[] {
  const groups = new Map<string, Clue[]>()

  for (const clue of clues) {
    if (!clue || typeof clue.value !== 'string' || !clue.value.trim()) continue
    const dimension = clue.dimension || 'other'
    const items = groups.get(dimension) ?? []
    items.push(clue)
    groups.set(dimension, items)
  }

  return [...groups.entries()]
    .map(([dimension, items]) => ({
      dimension,
      label: DIMENSION_LABELS[dimension] ?? dimension,
      items: dedupeByValue(items).sort((a, b) => safeConfidence(b) - safeConfidence(a)),
    }))
    .sort((a, b) => dimensionRank(a.dimension) - dimensionRank(b.dimension))
}

function dedupeByValue(items: readonly Clue[]): Clue[] {
  const best = new Map<string, Clue>()
  for (const item of items) {
    const key = item.value
    const current = best.get(key)
    if (!current || safeConfidence(item) > safeConfidence(current)) best.set(key, item)
  }
  return [...best.values()]
}

function safeConfidence(clue: Clue | undefined): number {
  return clue && Number.isFinite(clue.confidence) ? clue.confidence : 0
}

function dimensionRank(dimension: string): number {
  const index = DIMENSION_ORDER.indexOf(dimension)
  return index === -1 ? DIMENSION_ORDER.length : index
}

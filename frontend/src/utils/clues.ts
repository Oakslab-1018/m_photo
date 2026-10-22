import type { Clue } from '@/types'

export const DIMENSION_LABELS: Record<string, string> = {
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
}

export const VALUE_LABELS: Record<string, string> = {
  temperate_forest: '温带森林',
  grassland: '草原',
  savanna: '稀树草原',
  desert: '荒漠',
  tropical_rainforest: '热带雨林',
  urban_sparse: '城市稀疏植被',
  street_trees: '行道树',
  dark_vehicle: '深色车辆',
  white_silver_vehicle: '白/银色车辆',
  unknown_vehicle: '未识别车辆',
  buses: '公交车',
  bicycles: '自行车',
  motorcycles_scooters: '摩托车/踏板车',
  mediterranean: '地中海风格',
  colonial: '殖民风格',
  gothic: '哥特式',
  modern_glass: '现代玻璃幕墙',
  brick_masonry: '红砖/砖石建筑',
  high_rise: '高层建筑',
  low_rise: '低层建筑',
  traditional_architecture: '传统建筑',
  generic: '普通 / 不明显',
  mediterranean_sunny: '地中海晴朗气候',
  temperate_maritime: '温带海洋性气候',
  cold_polar: '寒冷气候',
  arid: '干旱气候',
  temperate: '温带气候',
  humid: '湿润气候',
  tropical: '热带气候',
  zh: '中文',
  ja: '日语',
  ko: '韩语',
  latin: '拉丁语系',
  cyrillic: '西里尔语系',
  arabic: '阿拉伯语',
  thai: '泰语',
  unknown: '未知',
  // driving_side（目录字段）
  left: '左行交通',
  right: '右行交通',
  // road（观察词条）
  street: '街道',
  road: '道路',
  avenue: '大道',
  boulevard: '林荫大道',
  highway: '高速公路',
  freeway: '高速公路',
  motorway: '高速公路',
  route: '路线',
  alley: '小巷',
  lane: '车道',
  square: '广场',
  plaza: '广场',
  '大道': '大道',
  '公路': '公路',
  '街道': '街道',
  '高速公路': '高速公路',
  '快速路': '快速路',
  '环路': '环路',
  // terrain（观察词条）
  mountain: '山地',
  hill: '丘陵',
  valley: '山谷',
  coast: '海岸',
  island: '岛屿',
  glacier: '冰川',
  volcano: '火山',
  canyon: '峡谷',
  plateau: '高原',
  cliff: '悬崖',
  '山脉': '山脉',
  '山谷': '山谷',
  '火山': '火山',
  '冰川': '冰川',
  '峡谷': '峡谷',
  '高原': '高原',
  '岛屿': '岛屿',
  '海岸': '海岸',
  // waterbody（观察词条）
  lake: '湖泊',
  river: '河流',
  sea: '海洋',
  ocean: '海洋',
  bay: '海湾',
  harbor: '港口',
  harbour: '港口',
  beach: '海滩',
  waterfall: '瀑布',
  reservoir: '水库',
  strait: '海峡',
  fjord: '峡湾',
  '湖泊': '湖泊',
  '河流': '河流',
  '海湾': '海湾',
  '海滩': '海滩',
  '瀑布': '瀑布',
  '水库': '水库',
  '海峡': '海峡',
  // infrastructure（观察词条）
  station: '车站',
  airport: '机场',
  port: '港口',
  bridge: '桥梁',
  tunnel: '隧道',
  dam: '水坝',
  metro: '地铁',
  subway: '地铁',
  stadium: '体育场',
  terminal: '航站楼',
  lighthouse: '灯塔',
  observatory: '天文台',
  university: '大学',
  hospital: '医院',
  museum: '博物馆',
  '车站': '车站',
  '机场': '机场',
  '港口': '港口',
  '大桥': '大桥',
  '隧道': '隧道',
  '水坝': '水坝',
  '地铁': '地铁',
  '体育场': '体育场',
  '博物馆': '博物馆',
  '大学': '大学',
  '医院': '医院',
  '灯塔': '灯塔',
  // culture（国家目录 culture 特征值）
  acropolis: '卫城',
  aztec: '阿兹特克',
  big_ben: '大本钟',
  cafe: '咖啡馆',
  castle: '城堡',
  colosseum: '罗马斗兽场',
  eiffel: '埃菲尔铁塔',
  elephant: '大象',
  flamenco: '弗拉门戈',
  hanbok: '韩服',
  kremlin: '克里姆林宫',
  lantern: '灯笼',
  maple: '枫叶',
  opera_house: '歌剧院',
  pagoda: '佛塔',
  pizza: '披萨',
  pyramid: '金字塔',
  red_bus: '红色巴士',
  sakura: '樱花',
  samba: '桑巴',
  sari: '纱丽',
  shrine: '神社',
  statue_of_liberty: '自由女神像',
  temple: '寺庙',
  viking: '维京',
  windmill: '风车',
}

const DIMENSION_ORDER = [
  'language',
  'place_name',
  'signage',
  'driving_side',
  'road',
  'terrain',
  'waterbody',
  'vegetation',
  'building',
  'vehicle',
  'climate',
  'culture',
  'infrastructure',
  'other',
]

/** Evidence channel / method behind one clue, used for honest provenance tags. */
export type ClueProvenance = 'user' | 'ocr' | 'observed' | 'model' | 'heuristic' | 'unknown'

const MODEL_SOURCES = new Set(['model', 'streetclip', 'deepseek', 'ml', 'cnn'])

const PROVENANCE_LABELS: Record<Exclude<ClueProvenance, 'unknown'>, string> = {
  user: '你补充的',
  ocr: '图片文字',
  observed: '区域分析',
  model: '图像分析',
  heuristic: '推断',
}

export interface ClueGroup {
  dimension: string
  label: string
  items: Clue[]
  /** Highest confidence inside the group, after dedupe. */
  maxConfidence: number
  /** True when the user explicitly supplied a clue for this dimension. */
  hasUserEvidence: boolean
}

export function dimensionLabel(dimension: string): string {
  return DIMENSION_LABELS[dimension] ?? dimension
}

export function clueValueLabel(value: string): string {
  return VALUE_LABELS[value] ?? value
}

/** Stable identity shared by guided snapshots and the left clue list. */
export function clueIdentity(dimension: string, value: string): string {
  return JSON.stringify([dimension, value])
}

export function confidencePercent(confidence: number): string {
  const safe = Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0
  return `${Math.round(safe * 100)}%`
}

export const LOW_CONFIDENCE_THRESHOLD = 0.35

/** Low (or unknown) confidence clues are visually de-emphasized, never hidden. */
export function isLowConfidence(confidence: number): boolean {
  return !Number.isFinite(confidence) || confidence < LOW_CONFIDENCE_THRESHOLD
}

export function clueProvenance(clue: Clue): ClueProvenance {
  const source = typeof clue.source === 'string' ? clue.source.trim().toLowerCase() : ''
  const origin =
    clue.extra && typeof clue.extra.origin_source === 'string'
      ? clue.extra.origin_source.trim().toLowerCase()
      : ''
  if (source === 'user' || origin === 'user') return 'user'
  if (source === 'ocr' || origin === 'ocr') return 'ocr'
  if (source === 'point_click') return 'observed'
  if (MODEL_SOURCES.has(source)) return 'model'
  if (source) return 'heuristic'
  return 'unknown'
}

/** Localized provenance label; any unrecognized named analyzer is treated as heuristic. */
export function clueSourceLabel(clue: Clue): string {
  const provenance = clueProvenance(clue)
  if (provenance !== 'unknown') return PROVENANCE_LABELS[provenance]
  const source = typeof clue.source === 'string' ? clue.source.trim() : ''
  return source
}

/** Backend reference-only marker: shown as an observation, never affecting ranking. */
export function isReferenceOnly(clue: Clue): boolean {
  const extra = clue.extra
  if (!extra || typeof extra !== 'object') return false
  return extra.display_only === true || extra.affects_ranking === false
}

/** Backend catalog-validation marker for clues checked against the country directory. */
export function isCatalogValidated(clue: Clue): boolean {
  return Boolean(clue.extra && typeof clue.extra === 'object' && clue.extra.validated === true)
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
    .map(([dimension, rawItems]) => {
      const items = dedupeByValue(rawItems).sort((a, b) => safeConfidence(b) - safeConfidence(a))
      return {
        dimension,
        label: dimensionLabel(dimension),
        items,
        maxConfidence: items.length ? safeConfidence(items[0]) : 0,
        hasUserEvidence: rawItems.some((item) => clueProvenance(item) === 'user'),
      }
    })
    .sort((a, b) => dimensionRank(a.dimension) - dimensionRank(b.dimension))
}

/** Number of clue cards actually shown after grouping/dedupe. */
export function countGroupedClues(clues: readonly Clue[]): number {
  return groupClues(clues).reduce((total, group) => total + group.items.length, 0)
}

/**
 * Dimensions that deserve to be expanded first: a user-supplied clue is the
 * most actionable evidence, otherwise the strongest observed clue wins.
 */
export function defaultOpenDimensions(groups: readonly ClueGroup[]): string[] {
  const userGroup = groups.find((group) => group.hasUserEvidence)
  if (userGroup) return [userGroup.dimension]
  let best: ClueGroup | null = null
  for (const group of groups) {
    if (!best || group.maxConfidence > best.maxConfidence) best = group
  }
  return best ? [best.dimension] : []
}

/** Cap long groups (e.g. one OCR line per signage clue) behind an expand control. */
export function cluePreview<T>(items: readonly T[], limit: number): { visible: T[]; hiddenCount: number } {
  if (!Number.isFinite(limit) || limit >= items.length) {
    return { visible: [...items], hiddenCount: 0 }
  }
  const size = Math.max(0, Math.floor(limit))
  return { visible: items.slice(0, size), hiddenCount: items.length - size }
}

function dedupeByValue(items: readonly Clue[]): Clue[] {
  const best = new Map<string, Clue>()
  for (const item of items) {
    const key = item.value
    const current = best.get(key)
    if (!current || preferClue(item, current)) best.set(key, item)
  }
  return [...best.values()]
}

/** A user-confirmed clue beats same-value machine evidence even at lower confidence. */
function preferClue(candidate: Clue, current: Clue): boolean {
  const candidateIsUser = clueProvenance(candidate) === 'user'
  const currentIsUser = clueProvenance(current) === 'user'
  if (candidateIsUser !== currentIsUser) return candidateIsUser
  return safeConfidence(candidate) > safeConfidence(current)
}

function safeConfidence(clue: Clue | undefined): number {
  return clue && Number.isFinite(clue.confidence) ? clue.confidence : 0
}

function dimensionRank(dimension: string): number {
  const index = DIMENSION_ORDER.indexOf(dimension)
  return index === -1 ? DIMENSION_ORDER.length : index
}
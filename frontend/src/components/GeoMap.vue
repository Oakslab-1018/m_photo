<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as maplibregl from 'maplibre-gl'
// Vector tiles are parsed in a Web Worker. MapLibre derives that worker's URL from
// import.meta.url, which after bundling points at this app's own chunk — a path where
// no worker file exists. Every tile parse then fails and the basemap collapses to the
// style's background color, while the map raises no fatal error and markers still
// draw, so it reads as "the tile server is down". Let Vite emit the worker instead
// (?worker&url keeps a .js extension, which Nginx serves as JavaScript) and point
// MapLibre at the hashed asset.
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import 'maplibre-gl/dist/maplibre-gl.css'
import type {
  CityScore,
  CountryScore,
  GuidedConfirmedPath,
  GuidedLocationLevel,
  GuidedStage,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import { cityPoints, countryPoints, guidedLocationPoints } from '@/utils/geoPoints'
import { locationLevelLabel } from '@/utils/guided'
import { resolveMapStyle } from '@/utils/mapConfig'
import {
  createMapLifecycle,
  type FitToPointsOptions,
  type MapLifecycle,
  type MapPoint,
} from '@/composables/useMapLibre'

// Idempotent, and the Map itself is only built in onMounted, so ordering is safe.
maplibregl.setWorkerUrl(maplibreWorkerUrl)

const props = withDefaults(
  defineProps<{
    scores: CountryScore[]
    confirmedIso: string | null
    cityScores?: CityScore[]
    cityCountryIso?: string | null
    selectedIso?: string | null
    /** Guided mode replaces the legacy score/city overlay with the server path/candidates. */
    guided?: boolean
    guidedStage?: GuidedStage | null
    guidedCandidates?: LocationCandidate[]
    /** False while the guided UI is still in the review phase (candidates hidden). */
    guidedCandidatesVisible?: boolean
    guidedConfirmed?: GuidedConfirmedPath | null
    selectedGuidedTargetKey?: string | null
    guidedConfirmationReady?: boolean
    busy?: boolean
  }>(),
  {
    cityScores: () => [],
    cityCountryIso: null,
    selectedIso: null,
    guided: false,
    guidedStage: null,
    guidedCandidates: () => [],
    guidedCandidatesVisible: true,
    guidedConfirmed: null,
    selectedGuidedTargetKey: null,
    guidedConfirmationReady: true,
    busy: false,
  },
)

const emit = defineEmits<{
  confirm: [score: CountryScore]
  'select-country': [iso: string]
  'select-guided-target': [target: LocationTarget]
  'confirm-guided-target': [target: LocationTarget]
}>()

const container = ref<HTMLElement | null>(null)
const mapError = ref<string | null>(null)
let lifecycle: MapLifecycle | null = null
let singleClickTimer: ReturnType<typeof setTimeout> | null = null

function cancelPendingMarkerClick(): void {
  if (singleClickTimer === null) return
  clearTimeout(singleClickTimer)
  singleClickTimer = null
}

/** Delay selection so the native second click of a double-click can cancel it. */
function scheduleMarkerClick(action: () => void): void {
  cancelPendingMarkerClick()
  singleClickTimer = setTimeout(() => {
    singleClickTimer = null
    action()
  }, 400)
}

const styleConfig = resolveMapStyle(import.meta.env as unknown as { VITE_MAP_STYLE_URL?: string; VITE_MAPTILER_API_KEY?: string })

const guidedMode = computed(() => props.guided)
const guidedMap = computed(() =>
  guidedLocationPoints(
    // Review phase hides every candidate marker; the confirmed path stays.
    props.guidedCandidatesVisible ? props.guidedCandidates : [],
    props.guidedConfirmed,
    props.guidedStage,
  ),
)
const countries = computed<MapPoint[]>(() => countryPoints(props.scores, props.confirmedIso))
const cities = computed<MapPoint[]>(() => cityPoints(props.cityScores))
const legacyPoints = computed<MapPoint[]>(() => [...countries.value, ...cities.value])
const displayPoints = computed<MapPoint[]>(() =>
  guidedMode.value ? guidedMap.value.points : legacyPoints.value,
)
const hasConfirmedPathPoints = computed(() =>
  guidedMap.value.points.some((point) => point.confirmed),
)
const candidateByIso = computed(() => {
  const lookup = new Map<string, CountryScore>()
  for (const score of props.scores) lookup.set(score.iso, score)
  return lookup
})

const guidedFrameSignature = computed(() => {
  if (!guidedMode.value) return ''
  const parts = guidedMap.value.points
    .map((point) =>
      [
        point.key,
        point.name,
        point.lat,
        point.lng,
        point.confirmed ? 'confirmed' : 'candidate',
        point.resolved ? 'resolved' : '',
      ].join(':'),
    )
    .sort()
  const missing = guidedMap.value.missingCoordinates
    .map((target) => target.level + ':' + target.id)
    .sort()
  return [props.guidedStage ?? '', ...parts, 'missing:' + missing.join(',')].join('|')
})

const guidedMarkerSignature = computed(() => {
  if (!guidedMode.value) return ''
  return guidedMap.value.points
    .map((point) =>
      [
        point.key,
        point.name,
        point.score ?? '',
        point.scoreType ?? '',
        point.coordinateSource ?? '',
        point.coordinateQuality ?? '',
      ].join(':'),
    )
    .join('|')
})

const guidedMissingNote = computed(() => {
  if (!guidedMode.value) return null
  const missing = guidedMap.value.missingCoordinates.length
  if (!missing) return null
  const parentCount = guidedMap.value.fallbackPoints.length
  if (parentCount) {
    return String(missing) + ' 个候选缺少有效目录坐标，未放置候选标记；选中后仅聚焦已确认父级范围'
  }
  return String(missing) + ' 个候选缺少有效目录坐标，未在地图放置标记'
})

const guidedSelectionNote = computed(() => {
  if (!guidedMode.value || !props.selectedGuidedTargetKey) return null
  const missing = guidedMap.value.missingCoordinates.find(
    (target) => target.level + ':' + target.id === props.selectedGuidedTargetKey,
  )
  if (!missing) return null
  const parent = nearestConfirmedParent()
  if (!parent) return '此候选无有效目录坐标，当前没有可聚焦的已确认父级范围'
  const parentLevel = parent.level ?? parent.kind
  return '此候选无坐标，当前显示父级范围（' + locationLevelLabel(parentLevel) + ' · ' + parent.name + '）'
})

const guidedMapNotice = computed(() => guidedSelectionNote.value ?? guidedMissingNote.value)

function pointIsSelected(point: MapPoint): boolean {
  if (guidedMode.value) return point.key === props.selectedGuidedTargetKey
  return point.kind === 'country' && point.iso === props.selectedIso
}

function scoreText(point: MapPoint): string {
  if (point.isCurrentCandidate && typeof point.score === 'number' && Number.isFinite(point.score)) {
    return ' · 排序分 ' + point.score.toFixed(3) + '（非概率）'
  }
  if (typeof point.cosine === 'number' && Number.isFinite(point.cosine)) {
    return ' · 余弦 ' + point.cosine.toFixed(3) + '（非概率）'
  }
  return ''
}
function isGuidedParentPoint(point: MapPoint): boolean {
  if (!point.confirmed) return false
  if (props.guidedStage === 'city') return point.level === 'admin1' || point.level === 'country'
  if (props.guidedStage === 'admin1') return point.level === 'country'
  return false
}

function guidedPointTooltip(point: MapPoint): string {
  const level = point.level ?? point.kind
  const levelLabel = locationLevelLabel(level)
  const score = scoreText(point)
  const admin1Provenance = point.derivedCoordinate
    ? '省级代表点（由目录最大城市坐标派生，不是行政区多边形，也不是照片 GPS）'
    : '省级目录代表点（不是行政区多边形，也不是照片 GPS）'

  if (point.resolved) {
    return '最终确认城市 ' + point.name + ' · 城市目录代表点，不是照片 GPS' + score
  }
  if (point.confirmed) {
    const parentPrefix = isGuidedParentPoint(point) ? '已确认父级' : '已确认'
    if (level === 'admin1') {
      return parentPrefix + '省份 / 地区 ' + point.name + ' · ' + admin1Provenance + score
    }
    return parentPrefix + levelLabel + ' ' + point.name + ' · 用户确认路径；坐标为代表点，不是照片 GPS' + score
  }
  const actionHint = props.guidedConfirmationReady
    ? '双击确认'
    : '完成新增观察核对后可双击确认'
  if (level === 'admin1') {
    return (
      '候选省份 / 地区 ' +
      point.name +
      ' · ' +
      admin1Provenance +
      score +
      ' · 单击选中，' +
      actionHint +
      '省份 / 地区'
    )
  }
  if (level === 'city') {
    return '候选城市 ' + point.name + ' · 城市目录代表点，不是照片 GPS' + score + ' · 单击选中，' + actionHint + '城市'
  }
  return (
    '候选国家 / 地区 ' +
    point.name +
    ' · 目录代表点，不是照片 GPS' +
    score +
    ' · 单击选中，' +
    actionHint +
    '国家 / 地区'
  )
}

function markerElement(point: MapPoint): HTMLElement {
  const el = document.createElement('button')
  el.type = 'button'
  const classes = ['maplibre-marker', 'maplibre-marker-' + point.kind]
  if (pointIsSelected(point)) classes.push('selected')
  if (point.confirmed) classes.push('confirmed')
  if (point.resolved) classes.push('resolved')
  if (point.isCurrentCandidate) classes.push('current-stage')
  if (!point.isCurrentCandidate && point.confirmed) classes.push('passive-parent')
  if (props.busy) classes.push('busy')
  el.className = classes.join(' ')
  el.disabled = props.busy
  el.setAttribute('data-target-key', point.key)
  el.setAttribute('data-kind', point.kind)
  if (point.level) el.setAttribute('data-level', point.level)

  const tooltip = guidedMode.value
    ? guidedPointTooltip(point)
    : point.kind === 'city'
      ? point.name + (point.subregion ? ' · ' + point.subregion : '') + ' · 城市代表点，不是照片位置' + scoreText(point)
      : point.name +
        (point.subregion ? '（区域命中：' + point.subregion + '）' : '') +
        (point.confirmed ? ' · 用户已确认的国家（非照片 GPS）' : '，单击选中，双击确认国家') +
        scoreText(point)
  const level = point.level ?? point.kind
  const ariaLevel = locationLevelLabel(level)
  el.setAttribute('aria-label', guidedMode.value ? tooltip : point.kind === 'city' ? '城市候选 ' + point.name : '候选国家 ' + point.name)
  el.title = tooltip

  if (guidedMode.value) {
    const target = point.target
    if (target && point.isCurrentCandidate) {
      el.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        if (props.busy) return
        scheduleMarkerClick(() => emit('select-guided-target', target))
      })
      if (point.level === props.guidedStage) {
        el.addEventListener('dblclick', (event) => {
          event.preventDefault()
          event.stopPropagation()
          if (
            props.busy ||
            !props.guidedConfirmationReady ||
            !point.isCurrentCandidate ||
            target.level !== props.guidedStage
          ) {
            return
          }
          cancelPendingMarkerClick()
          emit('confirm-guided-target', target)
        })
      }
    } else if (target && point.confirmed) {
      // Confirmed ancestors are focus-only. They can never confirm or advance a level.
      el.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        if (props.busy || !lifecycle) return
        cancelPendingMarkerClick()
        const level = point.level ?? point.kind
        lifecycle.focusPoint(point, { zoom: focusZoom(level, point.resolved === true), duration: 0 })
      })
    }
    el.setAttribute('aria-label', tooltip + '；层级：' + ariaLevel)
    return el
  }

  if (point.kind === 'country') {
    el.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      scheduleMarkerClick(() => emit('select-country', point.iso))
    })
    if (!point.confirmed) {
      el.addEventListener('dblclick', (event) => {
        event.preventDefault()
        event.stopPropagation()
        cancelPendingMarkerClick()
        const candidate = candidateByIso.value.get(point.iso)
        if (candidate) emit('confirm', candidate)
      })
    }
  }
  return el
}

function renderMarkers(): void {
  cancelPendingMarkerClick()
  lifecycle?.renderMarkers(displayPoints.value)
}
function stageFitOptions(): FitToPointsOptions {
  switch (props.guidedStage) {
    case 'country':
      return { padding: 56, maxZoom: 5, singleZoom: 4, duration: 0 }
    case 'admin1':
      return { padding: 56, maxZoom: 8, singleZoom: 7, duration: 0 }
    case 'city':
      return { padding: 56, maxZoom: 11, singleZoom: 10, duration: 0 }
    case 'resolved':
      return { padding: 56, maxZoom: 12, singleZoom: 12, duration: 0 }
    default:
      return { padding: 56, maxZoom: 6, singleZoom: 3, duration: 0 }
  }
}

function fallbackZoom(level: GuidedLocationLevel, resolved: boolean): number {
  if (resolved) return 12
  if (level === 'country') return 4
  if (level === 'admin1') return 6
  return 9
}

function focusZoom(level: GuidedLocationLevel, resolved: boolean): number {
  if (resolved) return 12
  if (level === 'country') return 4
  if (level === 'admin1') return 7
  return 10
}

function nearestConfirmedParent(): MapPoint | null {
  const stage = props.guidedStage
  const levels: GuidedLocationLevel[] =
    stage === 'city'
      ? ['admin1', 'country']
      : stage === 'admin1'
        ? ['country']
        : stage === 'resolved'
          ? ['city', 'admin1', 'country']
          : []
  for (const level of levels) {
    const point = guidedMap.value.points.find(
      (candidate) => candidate.confirmed && candidate.level === level,
    )
    if (point) return point
  }
  return null
}

function reframeGuided(): void {
  if (!lifecycle) return
  if (props.guidedStage === 'resolved') {
    const resolved = guidedMap.value.points.find((point) => point.resolved)
    const fallback = resolved ?? nearestConfirmedParent()
    if (fallback) lifecycle.focusPoint(fallback, { zoom: 12, duration: 0 })
    return
  }

  const candidates = guidedMap.value.fitPoints.filter((point) => point.isCurrentCandidate)
  if (candidates.length) {
    lifecycle.fitToPoints(candidates, stageFitOptions())
    return
  }

  const parent = nearestConfirmedParent()
  if (parent) {
    const level = parent.level ?? parent.kind
    lifecycle.focusPoint(parent, { zoom: fallbackZoom(level, false), duration: 0 })
  }
}

function reframeLegacy(): void {
  lifecycle?.fitToPoints(legacyPoints.value)
}

function focusGuidedSelection(): boolean {
  if (!guidedMode.value || !lifecycle) return false
  const key = props.selectedGuidedTargetKey
  if (!key) return false
  const point = displayPoints.value.find((candidate) => candidate.key === key)
  if (point) {
    const level = point.level ?? point.kind
    lifecycle.focusPoint(point, {
      zoom: focusZoom(level, point.resolved === true),
      duration: 0,
    })
    return true
  }
  const missing = guidedMap.value.missingCoordinates.find(
    (target) => target.level + ':' + target.id === key,
  )
  if (!missing) return false
  const parent = nearestConfirmedParent()
  if (!parent) return false
  const parentLevel = parent.level ?? parent.kind
  lifecycle.focusPoint(parent, { zoom: fallbackZoom(parentLevel, false), duration: 0 })
  return true
}

function focusCountries(): void {
  if (!guidedMode.value) lifecycle?.fitToPoints(countries.value)
}

function focusCities(): void {
  if (!guidedMode.value) lifecycle?.fitToPoints(cities.value)
}

function focusCurrentStage(): void {
  if (guidedMode.value) reframeGuided()
  else focusCountries()
}

function focusConfirmedPath(): void {
  if (!guidedMode.value) {
    focusCountries()
    return
  }
  const point = nearestConfirmedParent()
  if (!point) return
  const level = point.level ?? point.kind
  lifecycle?.focusPoint(point, { zoom: focusZoom(level, point.resolved === true), duration: 0 })
}

onMounted(() => {
  if (!container.value) return
  try {
    lifecycle = createMapLifecycle(maplibregl, {
      container: container.value,
      styleUrl: styleConfig.styleUrl,
      createMarkerElement: markerElement,
      onLoad: () => {
        mapError.value = null
      },
      onError: (event) => {
        mapError.value = '底图加载失败，请检查网络或地图样式配置'
        const error = (event as { error?: { message?: string } })?.error
        console.warn('MapLibre error', error?.message ?? error)
      },
    })
    renderMarkers()
    if (guidedMode.value) reframeGuided()
    else reframeLegacy()
  } catch (error) {
    mapError.value = '地图初始化失败，请检查浏览器 WebGL 支持'
    console.warn('MapLibre init failed', error)
  }
})

watch(
  [guidedFrameSignature, () => props.selectedGuidedTargetKey, guidedMode],
  ([nextFrame, nextKey, nextGuided], [previousFrame, previousKey, previousGuided]) => {
    if (!lifecycle) return
    renderMarkers()
    if (nextGuided !== previousGuided || nextFrame !== previousFrame) {
      if (nextGuided) reframeGuided()
      else reframeLegacy()
      return
    }
    if (nextGuided && nextKey !== previousKey) focusGuidedSelection()
  },
  { flush: 'post' },
)

// Score/reason-only updates redraw tooltips without changing the camera.
watch(guidedMarkerSignature, () => {
  if (guidedMode.value) renderMarkers()
})

watch(
  () => props.guidedConfirmationReady,
  () => {
    if (guidedMode.value) renderMarkers()
  },
)

watch(
  () => [props.scores, props.cityScores],
  () => {
    if (!guidedMode.value) {
      renderMarkers()
      reframeLegacy()
    }
  },
  { deep: false },
)

// Legacy selection/confirmation only changes highlighting, never the camera.
watch(
  () => [props.confirmedIso, props.selectedIso],
  () => {
    if (!guidedMode.value) renderMarkers()
  },
  { deep: false },
)

onBeforeUnmount(() => {
  cancelPendingMarkerClick()
  lifecycle?.dispose()
  lifecycle = null
})
</script>

<template>
  <div class="map-panel-inner">
    <div class="map-wrap">
      <div
        ref="container"
        class="maplibre-map"
        role="application"
        :aria-label="guidedMode ? '国家、省份与城市引导候选代表点交互地图' : '候选国家与城市代表点交互地图'"
      />
      <div v-if="styleConfig.demoOnly" class="map-badge demo">
        Demo 底图 · 仅开发预览，不可用于生产
      </div>
      <div v-if="guidedMapNotice" class="map-coordinate-note" role="status">
        {{ guidedMapNotice }}
      </div>
      <div v-if="!displayPoints.length" class="map-empty-overlay">
        {{ guidedMode ? '当前阶段暂无可显示的有效坐标候选' : '上传图片后，候选国家与城市代表点会显示在这里' }}
      </div>
      <div v-if="mapError" class="map-error-overlay" role="status">{{ mapError }}</div>
    </div>

    <div class="map-footer">
      <div v-if="guidedMode" class="map-legend">
        <span><i class="legend-dot" />国家 / 地区候选</span>
        <span><i class="legend-dot admin1" />省份代表点</span>
        <span><i class="legend-dot city" />城市候选</span>
        <span><i class="legend-dot confirmed" />已确认路径</span>
        <span v-if="guidedMap.fallbackPoints.length"><i class="legend-dot fallback" />已确认父级范围</span>
      </div>
      <div v-else class="map-legend">
        <span><i class="legend-dot" />候选国家</span>
        <span><i class="legend-dot city" />城市候选</span>
        <span><i class="legend-dot confirmed" />用户已确认国家</span>
      </div>
      <span class="map-hint">
        {{ guidedMode
          ? (guidedConfirmationReady
            ? '单击选中候选 · 双击确认当前层级 · 省级代表点非行政区多边形，所有坐标非照片 GPS'
            : '单击选中候选 · 新增观察待核对，暂不能确认地点 · 所有坐标非照片 GPS')
          : '单击选中国家 · 双击确认国家 · 标记均为代表点，不是照片 GPS' }}
      </span>
      <div class="focus-group">
        <template v-if="guidedMode">
          <button type="button" class="ghost-btn" :disabled="!displayPoints.length" @click="focusCurrentStage">
            聚焦当前候选
          </button>
          <button type="button" class="ghost-btn" :disabled="!hasConfirmedPathPoints" @click="focusConfirmedPath">
            聚焦已确认路径
          </button>
        </template>
        <template v-else>
          <button type="button" class="ghost-btn" :disabled="!countries.length" @click="focusCountries">聚焦国家</button>
          <button type="button" class="ghost-btn" :disabled="!cities.length" @click="focusCities">聚焦城市</button>
        </template>
      </div>
    </div>
  </div>
</template>

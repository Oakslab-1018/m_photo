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
import { guidedTargetKey, locationLevelLabel } from '@/utils/guided'
import { resolveMapStyle } from '@/utils/mapConfig'
import {
  createMapLifecycle,
  normalizeViewportInsets,
  type FitToPointsOptions,
  type MapLifecycle,
  type MapPoint,
  type ViewportInsets,
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
    /** Render the map legend/navigation as a safe-area overlay inside the canvas. */
    immersive?: boolean
    /** Floating UI coverage measured against the map canvas, not the page. */
    viewportInsets?: ViewportInsets
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
    immersive: false,
    viewportInsets: () => ({ top: 0, right: 0, bottom: 0, left: 0 }),
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

function retryBasemap(): void {
  try {
    if (!lifecycle?.retryStyle()) {
      mapError.value = '地图还没准备好，请稍后重试'
      return
    }
  } catch (error) {
    mapError.value = '地图加载失败，请稍后重试'
    console.warn('MapLibre retry failed', error)
  }
}

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
const normalizedViewportInsets = computed(() => normalizeViewportInsets(props.viewportInsets))
const viewportInsetStyle = computed<Record<string, string>>(() => ({
  '--geo-map-inset-top': normalizedViewportInsets.value.top + 'px',
  '--geo-map-inset-right': normalizedViewportInsets.value.right + 'px',
  '--geo-map-inset-bottom': normalizedViewportInsets.value.bottom + 'px',
  '--geo-map-inset-left': normalizedViewportInsets.value.left + 'px',
}))

const guidedMode = computed(() => props.guided)
const guidedMap = computed(() =>
  guidedLocationPoints(
    // Review phase hides every candidate marker; the confirmed path stays.
    props.guidedCandidatesVisible ? props.guidedCandidates : [],
    props.guidedConfirmed,
    props.guidedStage,
  ),
)
const guidedCandidateNumbers = computed(() => {
  const numbers = new Map<string, number>()
  props.guidedCandidates.forEach((candidate, index) => {
    const key = guidedTargetKey(candidate?.target)
    if (key && !numbers.has(key)) numbers.set(key, index + 1)
  })
  return numbers
})
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
        point.lat,
        point.lng,
        point.level ?? point.kind,
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
        point.isCurrentCandidate ? guidedCandidateNumber(point) ?? '' : '',
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
    return String(missing) + ' 个候选没有地图坐标，未显示标记；选中后会显示所属的已确认范围'
  }
  return String(missing) + ' 个候选没有地图坐标，未显示标记'
})

const guidedSelectionNote = computed(() => {
  if (!guidedMode.value || !props.selectedGuidedTargetKey) return null
  const missing = guidedMap.value.missingCoordinates.find(
    (target) => target.level + ':' + target.id === props.selectedGuidedTargetKey,
  )
  if (!missing) return null
  const parent = nearestConfirmedParent()
  if (!parent) return '这个候选没有地图坐标，暂时无法定位'
  const parentLevel = parent.level ?? parent.kind
  return '这个候选没有地图坐标，当前显示所属范围（' + locationLevelLabel(parentLevel) + ' · ' + parent.name + '）'
})

const guidedMapNotice = computed(() => guidedSelectionNote.value ?? guidedMissingNote.value)

function pointIsSelected(point: MapPoint): boolean {
  if (guidedMode.value) return point.key === props.selectedGuidedTargetKey
  return point.kind === 'country' && point.iso === props.selectedIso
}

function guidedCandidateNumber(point: MapPoint): number | null {
  if (!point.isCurrentCandidate) return null
  return guidedCandidateNumbers.value.get(point.key) ?? null
}

function scoreText(point: MapPoint): string {
  if (point.isCurrentCandidate && typeof point.score === 'number' && Number.isFinite(point.score)) {
    return ' · 匹配度 ' + point.score.toFixed(3) + '（不是概率）'
  }
  if (typeof point.cosine === 'number' && Number.isFinite(point.cosine)) {
    return ' · 图片相似度 ' + point.cosine.toFixed(3) + '（不是概率）'
  }
  return ''
}
function isGuidedParentPoint(point: MapPoint): boolean {
  return Boolean(
    point.confirmed && props.guidedStage === 'city' && point.level === 'country',
  )
}

function guidedPointTooltip(point: MapPoint): string {
  const level = point.level ?? point.kind
  const levelLabel = locationLevelLabel(level)
  const score = scoreText(point)

  if (point.resolved) {
    return '已确认城市 ' + point.name + ' · 位置是代表点，不是照片 GPS' + score
  }
  if (point.confirmed) {
    const parentPrefix = isGuidedParentPoint(point) ? '已确认的' : '已确认'
    return (
      parentPrefix +
      levelLabel +
      ' ' +
      point.name +
      ' · 位置是代表点，不是照片 GPS' +
      score
    )
  }
  const actionHint = props.guidedConfirmationReady
    ? '双击确认'
    : '核对新观察后才能确认'
  const candidateNumber = guidedCandidateNumber(point)
  const candidatePrefix = candidateNumber === null ? '候选' : '候选 ' + candidateNumber + ' · '
  if (level === 'city') {
    return candidatePrefix + '城市 ' + point.name + ' · 位置是代表点，不是照片 GPS' + score + ' · 单击选中，' + actionHint
  }
  return (
    candidatePrefix +
    '国家 / 地区 ' +
    point.name +
    ' · 位置是代表点，不是照片 GPS' +
    score +
    ' · 单击选中，' +
    actionHint
  )
}

function markerElement(point: MapPoint): HTMLElement {
  const el = document.createElement('button')
  el.type = 'button'
  const classes = ['maplibre-marker', 'geo-map-marker', 'maplibre-marker-' + point.kind]
  const candidateNumber = guidedCandidateNumber(point)
  if (pointIsSelected(point)) classes.push('selected')
  if (point.confirmed) classes.push('confirmed')
  if (point.resolved) classes.push('resolved')
  if (point.isCurrentCandidate) classes.push('current-stage')
  if (candidateNumber !== null) classes.push('numbered-candidate')
  if (!point.isCurrentCandidate && point.confirmed) classes.push('passive-parent')
  if (props.busy) classes.push('busy')
  el.className = classes.join(' ')
  el.disabled = props.busy
  el.setAttribute('data-target-key', point.key)
  el.setAttribute('data-kind', point.kind)
  el.setAttribute('data-selected', pointIsSelected(point) ? 'true' : 'false')
  el.setAttribute('data-confirmed', point.confirmed ? 'true' : 'false')
  if (point.level) el.setAttribute('data-level', point.level)
  if (candidateNumber !== null) el.setAttribute('data-candidate-number', String(candidateNumber))

  // Keep the button as a generous 38px hit target while the nested visual
  // remains crisp and compact. Native buttons preserve Enter/Space behavior.
  const visual = document.createElement('span')
  visual.className = 'geo-map-marker-visual'
  visual.setAttribute('aria-hidden', 'true')
  if (point.confirmed) {
    visual.textContent = '✓'
    visual.classList.add('confirmed-check')
  } else if (candidateNumber !== null) {
    visual.textContent = String(candidateNumber)
    visual.classList.add('candidate-index')
  }
  el.appendChild(visual)

  const tooltip = guidedMode.value
    ? guidedPointTooltip(point)
    : point.kind === 'city'
      ? point.name + (point.subregion ? ' · ' + point.subregion : '') + ' · 位置是代表点，不是照片 GPS' + scoreText(point)
      : point.name +
        (point.subregion ? '（区域：' + point.subregion + '）' : '') +
        (point.confirmed ? ' · 已确认（不是照片 GPS）' : '，单击选中，双击确认') +
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
  return 9
}

function focusZoom(level: GuidedLocationLevel, resolved: boolean): number {
  if (resolved) return 12
  if (level === 'country') return 4
  return 10
}

function nearestConfirmedParent(): MapPoint | null {
  const stage = props.guidedStage
  const levels: GuidedLocationLevel[] =
    stage === 'resolved' ? ['city', 'country'] : stage === 'city' ? ['country'] : []
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
      viewportInsets: normalizedViewportInsets.value,
      onError: (failure) => {
        mapError.value = failure.kind === 'source'
          ? '地图加载失败，请检查网络后重试'
          : '地图样式加载失败，请检查网络后重试'
        const error = failure.error as { message?: string } | undefined
        console.warn('MapLibre ' + failure.kind + ' error', error?.message ?? error)
      },
      onRecover: () => {
        mapError.value = null
      },
    })
    renderMarkers()
    if (guidedMode.value) reframeGuided()
    else reframeLegacy()
  } catch (error) {
    mapError.value = '地图无法显示，请确认浏览器支持 WebGL'
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
  () => props.busy,
  () => {
    renderMarkers()
  },
)

// Inset changes are a viewport-only adjustment. MapLibre keeps the geographic
// camera center/zoom while the visible center moves out from under the docks.
// Never reframe candidates here: a status/layout update must not reset a manual pan.
watch(
  normalizedViewportInsets,
  (nextInsets) => {
    lifecycle?.setViewportInsets(nextInsets)
  },
  { deep: true, flush: 'post' },
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
  <div
    class="map-panel-inner geo-map"
    :class="{ 'geo-map--immersive': immersive }"
    :style="viewportInsetStyle"
    data-geo-map
    :data-immersive="immersive ? 'true' : 'false'"
    :data-inset-top="normalizedViewportInsets.top"
    :data-inset-right="normalizedViewportInsets.right"
    :data-inset-bottom="normalizedViewportInsets.bottom"
    :data-inset-left="normalizedViewportInsets.left"
  >
    <div class="map-wrap geo-map__map-wrap" data-map-shell>
      <div
        ref="container"
        class="maplibre-map geo-map__canvas"
        data-map-canvas
        role="application"
        :aria-label="guidedMode ? '候选地图' : '候选国家与城市地图'"
      />
      <div v-if="styleConfig.demoOnly" class="map-badge demo geo-map__badge">
        Demo 底图 · 仅开发预览
      </div>
      <div class="geo-map__safe-center">
        <div v-if="!displayPoints.length" class="map-empty-overlay">
          {{ guidedMode ? '还没有可显示的地点' : '上传照片后显示候选地点' }}
        </div>
        <div v-if="mapError" class="map-error-overlay" role="status">
          <span>{{ mapError }}</span>
          <button
            type="button"
            class="ghost-btn map-error-overlay__retry"
            @click.stop="retryBasemap"
          >
            重试
          </button>
        </div>
      </div>
    </div>

    <div class="map-footer geo-map-footer">
      <div v-if="guidedMapNotice" class="map-coordinate-note geo-map-coordinate-note" role="status">
        {{ guidedMapNotice }}
      </div>
      <div v-if="guidedMode" class="map-legend">
        <span><i class="legend-dot" />国家 / 地区</span>
        <span><i class="legend-dot city" />城市</span>
        <span><i class="legend-dot confirmed" />已确认</span>
        <span v-if="guidedMap.fallbackPoints.length"><i class="legend-dot fallback" />已确认范围</span>
      </div>
      <div v-else class="map-legend">
        <span><i class="legend-dot" />候选国家</span>
        <span><i class="legend-dot city" />城市</span>
        <span><i class="legend-dot confirmed" />已确认</span>
      </div>
      <span class="map-hint">
        {{ guidedMode
          ? (guidedConfirmationReady
            ? '单击选中，双击确认 · 位置是代表点，不是照片 GPS'
            : '核对完成前不能确认 · 位置是代表点，不是照片 GPS')
          : '单击选中，双击确认 · 位置是代表点，不是照片 GPS' }}
      </span>
      <div class="focus-group">
        <template v-if="guidedMode">
          <button type="button" class="ghost-btn" :disabled="!displayPoints.length" @click="focusCurrentStage">
            显示当前候选
          </button>
          <button type="button" class="ghost-btn" :disabled="!hasConfirmedPathPoints" @click="focusConfirmedPath">
            显示已确认路径
          </button>
        </template>
        <template v-else>
          <button type="button" class="ghost-btn" :disabled="!countries.length" @click="focusCountries">显示国家</button>
          <button type="button" class="ghost-btn" :disabled="!cities.length" @click="focusCities">显示城市</button>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.geo-map.map-panel-inner {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: minmax(0, 1fr) auto;
  flex: 1;
  min-width: 0;
  min-height: 0;
}

.geo-map__map-wrap {
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
}

.geo-map-footer {
  grid-column: 1;
  grid-row: 2;
  min-width: 0;
}

.geo-map--immersive {
  grid-template-rows: minmax(0, 1fr);
}

.geo-map--immersive > .geo-map-footer {
  grid-column: 1;
  grid-row: 1;
  align-self: end;
  z-index: 20;
  box-sizing: border-box;
  max-height: calc(100% - var(--geo-map-inset-top) - var(--geo-map-inset-bottom) - 52px);
  /* Leave the compact attribution control its own unobstructed bottom row. */
  margin: 0 calc(var(--geo-map-inset-right) + 10px) calc(var(--geo-map-inset-bottom) + 40px) calc(var(--geo-map-inset-left) + 10px);
  padding: 6px 10px;
  overflow: auto;
  overscroll-behavior: contain;
  border: 1px solid var(--shell-line-strong);
  border-radius: 6px;
  background: rgba(9, 15, 18, 0.87);
  box-shadow: var(--shell-shadow-soft);
  backdrop-filter: blur(14px);
}

.geo-map--immersive > .geo-map-footer .ghost-btn {
  min-height: 27px;
  padding: 3px 8px;
  border-color: var(--shell-line-strong);
  background: transparent;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}

.geo-map--immersive > .geo-map-footer .ghost-btn:hover:not(:disabled) {
  border-color: var(--amber-line);
  color: var(--amber);
  background: var(--amber-soft);
}

.geo-map-coordinate-note {
  position: static;
  inset: auto;
  flex: 0 0 100%;
  width: 100%;
  max-width: none;
  box-sizing: border-box;
}

.geo-map__safe-center {
  position: absolute;
  inset: var(--geo-map-inset-top) var(--geo-map-inset-right) var(--geo-map-inset-bottom) var(--geo-map-inset-left);
  z-index: 10;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-sizing: border-box;
  padding: 10px;
  pointer-events: none;
}

.geo-map__safe-center .map-empty-overlay,
.geo-map__safe-center .map-error-overlay {
  position: static;
  inset: auto;
  transform: none;
  max-width: min(86%, 520px);
}

/* The safe-center layer is click-through, but the error action must stay usable. */
.geo-map__safe-center .map-error-overlay {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
  pointer-events: auto;
  text-align: center;
}

.geo-map__safe-center .map-error-overlay__retry {
  flex: 0 0 auto;
  min-height: 26px;
  padding: 2px 9px;
  pointer-events: auto;
}

.geo-map__badge {
  top: calc(var(--geo-map-inset-top) + 10px);
  left: calc(var(--geo-map-inset-left) + 10px);
  max-width: calc(100% - var(--geo-map-inset-left) - var(--geo-map-inset-right) - 20px);
  box-sizing: border-box;
}

/* Keep MapLibre's own controls out of the measured floating-dock area. */
.geo-map :deep(.maplibregl-ctrl-top-left) {
  top: var(--geo-map-inset-top);
  left: var(--geo-map-inset-left);
  z-index: 30;
}

.geo-map :deep(.maplibregl-ctrl-top-right) {
  top: var(--geo-map-inset-top);
  right: var(--geo-map-inset-right);
  z-index: 30;
}

.geo-map :deep(.maplibregl-ctrl-bottom-left) {
  bottom: var(--geo-map-inset-bottom);
  left: var(--geo-map-inset-left);
  z-index: 30;
}

.geo-map :deep(.maplibregl-ctrl-bottom-right) {
  right: var(--geo-map-inset-right);
  bottom: var(--geo-map-inset-bottom);
  z-index: 30;
}

/* Markers are created imperatively, so :deep reaches MapLibre's child nodes. */
.geo-map :deep(.maplibre-marker) {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  min-width: 44px;
  min-height: 44px;
  box-sizing: border-box;
  padding: 0;
  border: 0;
  border-radius: 999px;
  background: transparent;
  box-shadow: none;
  color: inherit;
  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
}

.geo-map :deep(.maplibre-marker.busy) {
  cursor: not-allowed;
}

.geo-map :deep(.maplibre-marker:focus-visible) {
  outline: 1.5px solid var(--amber);
  outline-offset: 3px;
  border-radius: 50%;
}

.geo-map :deep(.geo-map-marker-visual) {
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  box-sizing: border-box;
  border: 1.5px solid var(--amber);
  border-radius: 50%;
  background: #0a1114;
  color: var(--amber);
  box-shadow: 0 0 0 3px rgba(8, 13, 16, 0.55);
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  pointer-events: none;
}

.geo-map :deep(.maplibre-marker-city .geo-map-marker-visual) {
  width: 15px;
  height: 15px;
  border: 1.5px solid var(--blue);
  border-radius: 2px;
  background: #0a1114;
  color: var(--blue);
  font-size: 8px;
}

.geo-map :deep(.maplibre-marker.confirmed .geo-map-marker-visual) {
  border-color: var(--sage);
  background: var(--sage);
  color: #08120c;
}

.geo-map :deep(.maplibre-marker.resolved .geo-map-marker-visual) {
  border-color: var(--sage);
  background: var(--sage);
  color: #08120c;
  box-shadow:
    0 0 0 4px var(--sage-soft),
    0 0 0 3px rgba(8, 13, 16, 0.6);
}

.geo-map :deep(.maplibre-marker.passive-parent .geo-map-marker-visual) {
  border-style: dashed;
  border-color: var(--line-strong);
  color: var(--muted);
  box-shadow: 0 0 0 3px rgba(8, 13, 16, 0.5);
}

.geo-map :deep(.maplibre-marker.selected .geo-map-marker-visual) {
  border-color: var(--amber-bright);
  box-shadow:
    0 0 0 4px var(--amber-soft),
    0 0 0 3px rgba(8, 13, 16, 0.6);
}

.geo-map :deep(.maplibre-marker.selected .geo-map-marker-city .geo-map-marker-visual),
.geo-map :deep(.maplibre-marker-city.selected .geo-map-marker-visual) {
  border-color: var(--blue);
  background: var(--blue);
  color: #08120c;
}

.geo-map :deep(.geo-map-marker-visual.confirmed-check) {
  font-size: 12px;
  font-weight: 500;
}

@media (max-width: 760px) {
  .geo-map--immersive > .geo-map-footer {
    /* Attribution can wrap to two lines on a narrow screen. */
    margin-bottom: calc(var(--geo-map-inset-bottom) + 58px);
    gap: 7px;
    padding: 6px 8px;
    overflow-x: hidden;
  }

  .geo-map-footer .map-legend {
    flex-wrap: wrap;
    gap: 5px 10px;
    width: 100%;
  }

  .geo-map-footer .map-legend > span {
    white-space: nowrap;
  }

  .geo-map-footer .map-hint {
    flex: 1 1 100%;
    min-width: 0;
    line-height: 1.45;
  }

  .geo-map-footer .focus-group {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    width: 100%;
    margin-left: 0;
  }

  .geo-map--immersive > .geo-map-footer .ghost-btn {
    width: 100%;
    min-width: 0;
    min-height: 32px;
    padding-inline: 8px;
    font-size: 14px;
  font-weight: 400;
  line-height: 20px;
    white-space: normal;
  }
}

@media (hover: hover) and (prefers-reduced-motion: no-preference) {
  .geo-map :deep(.geo-map-marker-visual) {
    transition: transform 120ms ease, box-shadow 120ms ease;
  }

  .geo-map :deep(.maplibre-marker:not(:disabled):hover .geo-map-marker-visual) {
    transform: scale(1.08);
  }
}

@media (prefers-reduced-motion: reduce) {
  .geo-map :deep(.maplibre-marker),
  .geo-map :deep(.geo-map-marker-visual) {
    transition: none;
  }
}
</style>

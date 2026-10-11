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
import type { CityScore, CountryScore } from '@/types'
import { cityPoints, countryPoints } from '@/utils/geoPoints'
import { resolveMapStyle } from '@/utils/mapConfig'
import { createMapLifecycle, type MapLifecycle, type MapPoint } from '@/composables/useMapLibre'

// Idempotent, and the Map itself is only built in onMounted, so ordering is safe.
maplibregl.setWorkerUrl(maplibreWorkerUrl)

const props = defineProps<{
  scores: CountryScore[]
  confirmedIso: string | null
  cityScores?: CityScore[]
  cityCountryIso?: string | null
  selectedIso?: string | null
}>()

const emit = defineEmits<{
  confirm: [score: CountryScore]
  'select-country': [iso: string]
}>()

const container = ref<HTMLElement | null>(null)
const mapError = ref<string | null>(null)
let lifecycle: MapLifecycle | null = null

const styleConfig = resolveMapStyle(import.meta.env as unknown as { VITE_MAP_STYLE_URL?: string; VITE_MAPTILER_API_KEY?: string })

const countries = computed<MapPoint[]>(() => countryPoints(props.scores, props.confirmedIso))
const cities = computed<MapPoint[]>(() => cityPoints(props.cityScores))
const candidateByIso = computed(() => {
  const lookup = new Map<string, CountryScore>()
  for (const score of props.scores) lookup.set(score.iso, score)
  return lookup
})

function markerElement(point: MapPoint): HTMLElement {
  const el = document.createElement('button')
  el.type = 'button'
  el.className = point.kind === 'city' ? 'maplibre-marker maplibre-marker-city' : 'maplibre-marker'
  if (point.confirmed) el.classList.add('confirmed')
  if (point.kind === 'country' && point.iso === props.selectedIso) el.classList.add('selected')
  const cosine = point.cosine != null ? ` · 余弦 ${point.cosine.toFixed(3)}（非概率）` : ''
  // A CN candidate that aggregates regional evidence stays 中国; the tooltip
  // only adds the regional hit (中国台湾/中国香港/中国澳门).
  const scope = point.subregion ? `（区域命中：${point.subregion}）` : ''
  el.setAttribute(
    'aria-label',
    point.kind === 'city'
      ? `城市候选 ${point.name}${point.subregion ? `（${point.subregion}区域代表点，不是照片位置）` : '（代表点，不是照片位置）'}`
      : `候选国家 ${point.name}${scope}，单击选中，双击确认国家`,
  )
  el.title =
    point.kind === 'city'
      ? `${point.name}${point.subregion ? ` · ${point.subregion}` : ''} · 城市代表点，不是照片位置${cosine}`
      : `${point.name}${scope}${point.confirmed ? ' · 用户已确认的国家（非照片 GPS）' : '，单击选中，双击确认国家'}`
  if (point.kind === 'country') {
    el.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      emit('select-country', point.iso)
    })
    if (!point.confirmed) {
      el.addEventListener('dblclick', (event) => {
        event.preventDefault()
        event.stopPropagation()
        const candidate = candidateByIso.value.get(point.iso)
        if (candidate) emit('confirm', candidate)
      })
    }
  }
  return el
}

function renderMarkers(): void {
  if (!lifecycle) return
  lifecycle.renderMarkers([...countries.value, ...cities.value])
}

/** New candidates/cities: re-place the markers and reframe the camera on them. */
function refresh(): void {
  if (!lifecycle) return
  const points = [...countries.value, ...cities.value]
  lifecycle.renderMarkers(points)
  lifecycle.fitToPoints(points)
}

function focusCountries(): void {
  lifecycle?.fitToPoints(countries.value)
}

function focusCities(): void {
  lifecycle?.fitToPoints(cities.value)
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
    refresh()
  } catch (error) {
    mapError.value = '地图初始化失败，请检查浏览器 WebGL 支持'
    console.warn('MapLibre init failed', error)
  }
})

watch(() => [props.scores, props.cityScores], refresh, { deep: false })
// Selecting or confirming only changes highlighting, not the data: redraw the markers
// and leave the camera alone. Refitting here would yank a user who had just zoomed
// into an area straight back out to the full extent every time they pick a candidate.
watch(() => [props.confirmedIso, props.selectedIso], renderMarkers, { deep: false })

onBeforeUnmount(() => {
  lifecycle?.dispose()
  lifecycle = null
})
</script>

<template>
  <div class="map-panel-inner">
    <div class="map-wrap">
      <div ref="container" class="maplibre-map" role="application" aria-label="候选国家与城市代表点交互地图" />
      <div v-if="styleConfig.demoOnly" class="map-badge demo">
        Demo 底图 · 仅开发预览，不可用于生产
      </div>
      <div v-if="!countries.length && !cities.length" class="map-empty-overlay">
        上传图片后，候选国家与城市代表点会显示在这里
      </div>
      <div v-if="mapError" class="map-error-overlay" role="status">{{ mapError }}</div>
    </div>

    <div class="map-footer">
      <div class="map-legend">
        <span><i class="legend-dot" />候选国家</span>
        <span><i class="legend-dot city" />城市候选</span>
        <span><i class="legend-dot confirmed" />用户已确认国家</span>
      </div>
      <span class="map-hint">单击选中国家 · 双击确认国家 · 标记均为代表点，不是照片 GPS</span>
      <div class="focus-group">
        <button type="button" class="ghost-btn" :disabled="!countries.length" @click="focusCountries">聚焦国家</button>
        <button type="button" class="ghost-btn" :disabled="!cities.length" @click="focusCities">聚焦城市</button>
      </div>
    </div>
  </div>
</template>

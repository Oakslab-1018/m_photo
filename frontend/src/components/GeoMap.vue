<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { CountryScore } from '@/types'
import { useMapViewport, toMapMarkers } from '@/composables/useMapViewport'
import { scoreLabel } from '@/utils/map'

const props = defineProps<{
  scores: CountryScore[]
  confirmedIso: string | null
  exclusions: string[]
  busy: boolean
}>()

const emit = defineEmits<{
  confirm: [score: CountryScore]
  reject: [score: CountryScore]
}>()

const viewport = useMapViewport()
const mapRef = ref<HTMLElement | null>(null)
const imageRef = ref<HTMLImageElement | null>(null)
const markers = computed(() => toMapMarkers(props.scores, props.confirmedIso))

function recenter(): void {
  viewport.center(mapRef.value?.clientHeight ?? 0, imageRef.value?.offsetHeight ?? 0)
}

onMounted(() => {
  window.addEventListener('resize', recenter)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', recenter)
})
</script>

<template>
  <div class="map-panel-inner">
    <div
      ref="mapRef"
      class="map"
      :class="{ grabbing: viewport.dragging.value }"
      @mousedown="viewport.onMouseDown"
      @touchstart="viewport.onTouchStart"
      @touchmove="viewport.onTouchMove"
      @touchend="viewport.onTouchEnd"
      @wheel="viewport.onWheel"
    >
      <div
        class="map-canvas"
        :style="{
          transform: viewport.viewportTransform.value,
          '--map-scale': viewport.scale.value,
        }"
      >
        <img
          ref="imageRef"
          src="/world_map.png"
          class="world-map-img"
          alt="世界地图，范围为经度负180到正180、纬度负60到正80"
          draggable="false"
          @load="recenter"
        >
        <div class="map-markers">
          <button
            v-for="marker in markers"
            :key="marker.score.iso"
            type="button"
            class="map-marker"
            :class="{ top: marker.rank === 0, confirmed: confirmedIso === marker.score.iso }"
            :style="{
              left: marker.left,
              top: marker.top,
              width: marker.size,
              height: marker.size,
              background: marker.color,
            }"
            :title="marker.title"
            @dblclick="emit('confirm', marker.score)"
          >
            <span class="rank-num">{{ marker.rank + 1 }}</span>
            <span class="marker-label">{{ marker.score.name_zh }} {{ marker.scoreText }}</span>
          </button>
        </div>
      </div>
      <div v-if="!scores.length" class="map-empty">上传图片后，候选国家会在这里高亮显示</div>
      <div class="map-tip">拖拽平移 · 滚轮缩放 · 双击标记确认</div>
    </div>

    <div class="candidate-panel">
      <div class="candidate-title">候选国家</div>
      <div v-if="!scores.length" class="empty-hint">暂无候选</div>
      <div v-else class="candidate-list">
        <article v-for="(candidate, index) in scores" :key="candidate.iso" class="candidate-card">
          <div class="candidate-main">
            <span class="candidate-rank">{{ index + 1 }}</span>
            <div>
              <div class="candidate-name">{{ candidate.name_zh }}</div>
              <div class="candidate-score">{{ scoreLabel(candidate) }}</div>
            </div>
          </div>
          <div class="candidate-actions">
            <button type="button" class="mini-btn confirm" :disabled="busy" @click="emit('confirm', candidate)">确认</button>
            <button type="button" class="mini-btn reject" :disabled="busy" @click="emit('reject', candidate)">排除</button>
          </div>
        </article>
      </div>
      <div v-if="exclusions.length" class="exclusion-note">已排除 ISO：{{ exclusions.join('、') }}</div>
    </div>
  </div>
</template>

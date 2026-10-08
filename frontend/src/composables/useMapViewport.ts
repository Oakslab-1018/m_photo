import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { CountryScore } from '@/types'
import { latLngToPercent, markerColor, markerSize, scoreLabel } from '@/utils/map'

const MIN_SCALE = 1
const MAX_SCALE = 4

export interface MapMarkerView {
  score: CountryScore
  rank: number
  left: string
  top: string
  size: string
  color: string
  scoreText: string
  title: string
}

/** Pan/zoom state for the static world map viewport. */
export function useMapViewport() {
  const scale = ref(1)
  const offsetX = ref(0)
  const offsetY = ref(0)
  const dragging = ref(false)

  let startX = 0
  let startY = 0
  let originX = 0
  let originY = 0

  const viewportTransform = computed(
    () => `translate(${offsetX.value}px, ${offsetY.value}px) scale(${scale.value})`,
  )
  const inverseScale = computed(() => 1 / scale.value)

  function applyZoom(delta: number, anchorX: number, anchorY: number): void {
    const nextScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale.value * delta))
    if (nextScale === scale.value) return
    const worldX = (anchorX - offsetX.value) / scale.value
    const worldY = (anchorY - offsetY.value) / scale.value
    scale.value = nextScale
    offsetX.value = anchorX - worldX * nextScale
    offsetY.value = anchorY - worldY * nextScale
  }

  function center(containerHeight: number, imageHeight: number): void {
    scale.value = 1
    offsetX.value = 0
    offsetY.value = Math.max(0, (containerHeight - imageHeight) / 2)
  }

  function startDrag(clientX: number, clientY: number): void {
    dragging.value = true
    startX = clientX
    startY = clientY
    originX = offsetX.value
    originY = offsetY.value
  }

  function moveDrag(clientX: number, clientY: number): void {
    if (!dragging.value) return
    offsetX.value = clampOffset(originX + clientX - startX)
    offsetY.value = clampOffset(originY + clientY - startY)
  }

  function endDrag(): void {
    dragging.value = false
  }

  function onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return
    startDrag(event.clientX, event.clientY)
    event.preventDefault()
  }

  function onMouseMove(event: MouseEvent): void {
    if (!dragging.value) return
    moveDrag(event.clientX, event.clientY)
  }

  function onTouchStart(event: TouchEvent): void {
    const touch = event.touches[0]
    if (touch) startDrag(touch.clientX, touch.clientY)
  }

  function onTouchMove(event: TouchEvent): void {
    const touch = event.touches[0]
    if (touch) moveDrag(touch.clientX, touch.clientY)
  }

  function onWheel(event: WheelEvent): void {
    event.preventDefault()
    const target = event.currentTarget as HTMLElement
    const rect = target.getBoundingClientRect()
    const delta = event.deltaY < 0 ? 1.15 : 1 / 1.15
    applyZoom(delta, event.clientX - rect.left, event.clientY - rect.top)
  }

  onMounted(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', endDrag)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mouseup', endDrag)
  })

  return {
    scale,
    inverseScale,
    dragging,
    viewportTransform,
    center,
    onMouseDown,
    onTouchStart,
    onTouchMove,
    onTouchEnd: endDrag,
    onWheel,
  }
}

function clampOffset(value: number): number {
  return Math.max(-800, Math.min(800, value))
}

/** Convert ranked country scores into declarative map marker views. */
export function toMapMarkers(
  scores: readonly CountryScore[],
  confirmedIso: string | null,
): MapMarkerView[] {
  return scores.map((score, rank) => {
    const position = latLngToPercent(score.lat, score.lng)
    const confirmed = confirmedIso === score.iso
    return {
      score,
      rank,
      left: `${position.x}%`,
      top: `${position.y}%`,
      size: `${markerSize(score.score)}px`,
      color: confirmed ? '#ef4444' : markerColor(rank),
      scoreText: scoreLabel(score),
      title: `${score.name_zh}（${scoreLabel(score)}）${confirmed ? '，已确认拍摄地' : '，双击确认'}`,
    }
  })
}

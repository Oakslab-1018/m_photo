import { computed, onBeforeUnmount, onMounted, ref, type Ref } from 'vue'

export type LayoutBreakpoint = 'mobile' | 'compact' | 'desktop'

export interface ViewportInsets {
  top: number
  right: number
  bottom: number
  left: number
}

export interface LayoutRect {
  top: number
  right: number
  bottom: number
  left: number
  width: number
  height: number
}

export interface ViewportInsetsInput {
  host: LayoutRect
  topOverlay?: LayoutRect | null
  leftDock?: LayoutRect | null
  rightDock?: LayoutRect | null
  bottomOverlay?: LayoutRect | null
  showTop?: boolean
  showLeft?: boolean
  showRight?: boolean
  showBottom?: boolean
  clearance?: number
}

export interface ImmersiveLayoutOptions {
  commandBar: Ref<HTMLElement | null>
  mapHost: Ref<HTMLElement | null>
  leftDock: Ref<HTMLElement | null>
  rightDock: Ref<HTMLElement | null>
  includeTop: () => boolean
  includeLeft: () => boolean
  includeRight: () => boolean
  includeBottom: () => boolean
}

export const MIN_SAFE_MAP_EXTENT = 96

const ZERO_INSETS: ViewportInsets = { top: 0, right: 0, bottom: 0, left: 0 }
const MOBILE_QUERY = '(max-width: 760px)'
const COMPACT_QUERY = '(max-width: 1280px), (max-height: 760px)'

function finiteRect(rect: DOMRect | null | undefined): LayoutRect | null {
  if (!rect || !Number.isFinite(rect.left) || !Number.isFinite(rect.top)) return null
  return {
    top: rect.top,
    right: rect.right,
    bottom: rect.bottom,
    left: rect.left,
    width: rect.width,
    height: rect.height,
  }
}

function clampInset(value: number, limit: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(limit) || limit <= 0) return 0
  return Math.round(Math.min(limit, Math.max(0, value)))
}

/** Preserve a positive map window even when a full sheet and command bar overlap. */
function constrainPair(start: number, end: number, extent: number): [number, number] {
  const minimum = Math.min(MIN_SAFE_MAP_EXTENT, Math.max(1, Math.floor(extent)))
  const maximumCombined = Math.max(0, extent - minimum)
  if (start + end <= maximumCombined) return [start, end]
  const safeStart = Math.min(start, maximumCombined)
  return [safeStart, Math.min(end, Math.max(0, maximumCombined - safeStart))]
}

/**
 * Convert actual floating-surface geometry into map viewport insets.
 * Values are measured against the map container's own rectangle, not the page.
 */
export function calculateViewportInsets(input: ViewportInsetsInput): ViewportInsets {
  const { host } = input
  if (
    !Number.isFinite(host.left) ||
    !Number.isFinite(host.top) ||
    !Number.isFinite(host.width) ||
    !Number.isFinite(host.height) ||
    host.width <= 0 ||
    host.height <= 0
  ) {
    return { ...ZERO_INSETS }
  }

  const clearance = Number.isFinite(input.clearance) ? Math.max(0, input.clearance ?? 0) : 12
  const top = input.showTop && input.topOverlay
    ? clampInset(input.topOverlay.bottom - host.top + clearance, host.height)
    : 0
  const left = input.showLeft && input.leftDock && input.leftDock.width > 0 && input.leftDock.height > 0
    ? clampInset(input.leftDock.right - host.left + clearance, host.width)
    : 0
  const right = input.showRight && input.rightDock && input.rightDock.width > 0 && input.rightDock.height > 0
    ? clampInset(host.right - input.rightDock.left + clearance, host.width)
    : 0
  const bottom = input.showBottom && input.bottomOverlay && input.bottomOverlay.width > 0 && input.bottomOverlay.height > 0
    ? clampInset(host.bottom - input.bottomOverlay.top + clearance, host.height)
    : 0

  const [safeTop, safeBottom] = constrainPair(top, bottom, host.height)
  const [safeLeft, safeRight] = constrainPair(left, right, host.width)
  return { top: safeTop, right: safeRight, bottom: safeBottom, left: safeLeft }
}

/**
 * Track command-bar/dock/sheet geometry and responsive breakpoints. Observing
 * the real DOM keeps collapsed chips and changing mobile sheet heights honest.
 */
export function useImmersiveLayout(options: ImmersiveLayoutOptions) {
  const mapInsets = ref<ViewportInsets>({ ...ZERO_INSETS })
  const breakpoint = ref<LayoutBreakpoint>('desktop')
  const isMobile = computed(() => breakpoint.value === 'mobile')
  const isCompact = computed(() => breakpoint.value !== 'desktop')
  let resizeObserver: ResizeObserver | null = null
  let frame: number | null = null
  let mobileQuery: MediaQueryList | null = null
  let compactQuery: MediaQueryList | null = null

  function readBreakpoint(): void {
    if (typeof window === 'undefined') return
    breakpoint.value = mobileQuery?.matches
      ? 'mobile'
      : compactQuery?.matches
        ? 'compact'
        : 'desktop'
  }

  function recompute(): void {
    frame = null
    if (typeof window === 'undefined') return
    const hostEl = options.mapHost.value
    if (!hostEl) {
      mapInsets.value = { ...ZERO_INSETS }
      return
    }
    const host = finiteRect(hostEl.getBoundingClientRect())
    if (!host) return
    mapInsets.value = calculateViewportInsets({
      host,
      topOverlay: finiteRect(options.commandBar.value?.getBoundingClientRect()),
      leftDock: finiteRect(options.leftDock.value?.getBoundingClientRect()),
      rightDock: finiteRect(options.rightDock.value?.getBoundingClientRect()),
      bottomOverlay: finiteRect(options.rightDock.value?.getBoundingClientRect()),
      showTop: options.includeTop(),
      showLeft: breakpoint.value !== 'mobile' && options.includeLeft(),
      showRight: breakpoint.value !== 'mobile' && options.includeRight(),
      showBottom: breakpoint.value === 'mobile' && options.includeBottom(),
      clearance: 12,
    })
  }

  function scheduleRecompute(): void {
    if (typeof window === 'undefined' || frame !== null) return
    frame = window.requestAnimationFrame(recompute)
  }

  function onViewportChange(): void {
    readBreakpoint()
    scheduleRecompute()
  }

  onMounted(() => {
    if (typeof window === 'undefined') return
    mobileQuery = window.matchMedia(MOBILE_QUERY)
    compactQuery = window.matchMedia(COMPACT_QUERY)
    readBreakpoint()
    mobileQuery.addEventListener?.('change', onViewportChange)
    compactQuery.addEventListener?.('change', onViewportChange)
    window.addEventListener('resize', onViewportChange)
    window.addEventListener('orientationchange', scheduleRecompute)

    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(scheduleRecompute)
      for (const element of [
        options.commandBar.value,
        options.mapHost.value,
        options.leftDock.value,
        options.rightDock.value,
      ]) {
        if (element) resizeObserver.observe(element)
      }
    }
    scheduleRecompute()
  })

  onBeforeUnmount(() => {
    if (typeof window === 'undefined') return
    if (frame !== null) window.cancelAnimationFrame(frame)
    frame = null
    mobileQuery?.removeEventListener?.('change', onViewportChange)
    compactQuery?.removeEventListener?.('change', onViewportChange)
    window.removeEventListener('resize', onViewportChange)
    window.removeEventListener('orientationchange', scheduleRecompute)
    resizeObserver?.disconnect()
    resizeObserver = null
  })

  return {
    mapInsets,
    breakpoint,
    isMobile,
    isCompact,
  }
}
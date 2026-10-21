import { boundsForPoints, type LngLatBoundsLike } from '@/utils/geoPoints'
import type { GuidedLocationLevel, LocationTarget } from '@/types'

/**
 * Minimal structural types for the MapLibre API surface this app uses. Keeping
 * them structural lets tests drive the lifecycle with a fake namespace in a
 * plain node environment (no WebGL, no DOM mounting).
 */
export interface MapLike {
  on(type: string, listener: (event: unknown) => void): unknown
  addControl(control: unknown, position?: string): unknown
  fitBounds(bounds: [[number, number], [number, number]], options?: Record<string, unknown>): unknown
  easeTo(options: Record<string, unknown>): unknown
  setPadding(padding: ViewportInsets): unknown
  setStyle(style: string, options?: { diff?: boolean }): unknown
  remove(): void
}

export interface MarkerLike {
  setLngLat(lngLat: [number, number]): MarkerLike
  addTo(map: MapLike): MarkerLike
  remove(): void
}

export interface MapLibreNamespace {
  // Accepts MapLibre's concrete MapOptions/MarkerOptions while keeping the
  // structural surface we exercise in tests.
  Map: new (options: never) => MapLike
  Marker: new (options: never) => MarkerLike
  NavigationControl: new (options?: never) => unknown
}

export interface MapPoint {
  key: string
  kind: GuidedLocationLevel
  iso: string
  name: string
  label?: string
  lat: number
  lng: number
  cosine?: number | null
  scoreType?: string | null
  confirmed?: boolean
  /** Optional region scope evidence (中国台湾/中国香港/中国澳门) for tooltips. */
  subregion?: string | null
  /** Guided target backing this marker. Absent for legacy country/city rows. */
  target?: LocationTarget
  level?: GuidedLocationLevel
  score?: number | null
  coordinateSource?: string | null
  coordinateQuality?: string | null
  isCurrentCandidate?: boolean
  resolved?: boolean
  derivedCoordinate?: boolean
}

export interface FitToPointsOptions {
  /** Extra breathing room inside the current viewport insets. */
  padding?: number
  maxZoom?: number
  /** Zoom used for a single point; singlePointZoom is an accepted alias. */
  singleZoom?: number
  singlePointZoom?: number
  duration?: number
}

export interface FocusPointOptions {
  zoom?: number
  duration?: number
}

/** Screen-space area covered by floating UI, measured against the map canvas. */
export interface ViewportInsets {
  top: number
  right: number
  bottom: number
  left: number
}

export interface MapLifecycleFailure {
  /** Source failures are recoverable once that exact source proves a tile loaded. */
  kind: 'source' | 'style'
  error: unknown
  sourceId?: string
}

export interface MapLifecycleOptions {
  container: HTMLElement
  styleUrl: string
  createMarkerElement: (point: MapPoint) => HTMLElement
  /** Initial overlay safe area. Can be updated without changing the camera. */
  viewportInsets?: ViewportInsets
  onLoad?: () => void
  onError?: (failure: MapLifecycleFailure) => void
  /** Called only after every recorded failure has independent recovery evidence. */
  onRecover?: () => void
}

export interface MapLifecycle {
  map: MapLike
  renderMarkers(points: readonly MapPoint[]): void
  fitToPoints(points: readonly MapPoint[], options?: FitToPointsOptions): void
  focusPoint(point: MapPoint, options?: FocusPointOptions): void
  setViewportInsets(insets: ViewportInsets): void
  getViewportInsets(): ViewportInsets
  markers(): readonly MarkerLike[]
  /** Reload the same style while retaining camera, padding, and marker instances. */
  retryStyle(): boolean
  dispose(): void
}

function normalizedInset(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0
}

/** Normalize an untrusted DOM measurement before handing it to MapLibre. */
export function normalizeViewportInsets(
  insets: Partial<ViewportInsets> | null | undefined,
): ViewportInsets {
  return {
    top: normalizedInset(insets?.top ?? 0),
    right: normalizedInset(insets?.right ?? 0),
    bottom: normalizedInset(insets?.bottom ?? 0),
    left: normalizedInset(insets?.left ?? 0),
  }
}

function sameInsets(left: ViewportInsets, right: ViewportInsets): boolean {
  return (
    left.top === right.top &&
    left.right === right.right &&
    left.bottom === right.bottom &&
    left.left === right.left
  )
}

/** Build the MapLibre map plus marker/control lifecycle used by GeoMap.vue. */
export function createMapLifecycle(
  maplibre: MapLibreNamespace,
  options: MapLifecycleOptions,
): MapLifecycle {
  const MapCtor = maplibre.Map as unknown as new (options: Record<string, unknown>) => MapLike
  const MarkerCtor = maplibre.Marker as unknown as new (options: { element: HTMLElement }) => MarkerLike
  const NavigationCtor = maplibre.NavigationControl as unknown as new (
    options?: Record<string, unknown>,
  ) => unknown
  const map = new MapCtor({
    container: options.container,
    style: options.styleUrl,
    center: [0, 20],
    zoom: 1,
    attributionControl: { compact: true },
    trackResize: true,
  })
  map.addControl(new NavigationCtor({ visualizePitch: true }), 'top-right')

  let active: MarkerLike[] = []
  let disposed = false
  let viewportInsets = normalizeViewportInsets(options.viewportInsets)
  const failedSources = new Set<string>()
  let styleFailed = false
  let failureVisible = false

  function reportFailure(failure: MapLifecycleFailure): void {
    failureVisible = true
    options.onError?.(failure)
  }

  function recoverIfHealthy(): void {
    if (!failureVisible || styleFailed || failedSources.size) return
    failureVisible = false
    options.onRecover?.()
  }

  map.on('error', (rawEvent) => {
    if (disposed) return
    const event = rawEvent as { error?: unknown; sourceId?: unknown } | null
    const sourceId = typeof event?.sourceId === 'string' && event.sourceId ? event.sourceId : null
    const error = event?.error ?? rawEvent
    if (sourceId) {
      failedSources.add(sourceId)
      reportFailure({ kind: 'source', sourceId, error })
      return
    }
    styleFailed = true
    reportFailure({ kind: 'style', error })
  })

  // A loaded tile is the only recovery proof. MapLibre also emits load/idle after
  // sources with failed tiles settle, and metadata/other-source events do not
  // prove that the failed source is usable again.
  map.on('sourcedata', (rawEvent) => {
    if (disposed) return
    const event = rawEvent as { sourceId?: unknown; tile?: { state?: unknown } | null } | null
    const sourceId = typeof event?.sourceId === 'string' ? event.sourceId : null
    if (!sourceId || event?.tile?.state !== 'loaded') return
    if (!failedSources.delete(sourceId)) return
    recoverIfHealthy()
  })

  // A style failure is independent of source-tile recovery. A later successful
  // style load (for example after retryStyle) is its recovery evidence.
  map.on('style.load', () => {
    if (disposed || !styleFailed) return
    styleFailed = false
    recoverIfHealthy()
  })

  map.on('load', () => {
    if (disposed) return
    options.onLoad?.()
  })

  // MapLibre's edge insets shift the vanishing point without changing the
  // geographic camera center. Keeping this as the map's persistent padding lets
  // fitBounds add its own content margin exactly once (padding is additive by
  // default) and lets inset updates preserve manual pans and zoom.
  map.setPadding({ ...viewportInsets })

  function renderMarkers(points: readonly MapPoint[]): void {
    if (disposed) return
    for (const marker of active) marker.remove()
    active = points.map((point) =>
      new MarkerCtor({ element: options.createMarkerElement(point) })
        .setLngLat([point.lng, point.lat])
        .addTo(map),
    )
  }

  function fitToPoints(points: readonly MapPoint[], options: FitToPointsOptions = {}): void {
    if (disposed) return
    const bounds: LngLatBoundsLike | null = boundsForPoints(points)
    if (!bounds) return
    const padding = options.padding ?? 48
    const maxZoom = options.maxZoom ?? 6
    const duration = options.duration ?? 0
    if (bounds.west === bounds.east && bounds.south === bounds.north) {
      const zoom = options.singlePointZoom ?? options.singleZoom ?? 3
      map.easeTo({
        center: [bounds.west, bounds.south],
        zoom,
        duration,
        padding: { ...viewportInsets },
      })
      return
    }
    map.fitBounds(
      [
        [bounds.west, bounds.south],
        [bounds.east, bounds.north],
      ],
      // MapLibre adds this value to the persistent viewport padding above.
      { padding, maxZoom, duration, absolutePadding: false },
    )
  }

  function focusPoint(point: MapPoint, options: FocusPointOptions = {}): void {
    if (disposed) return
    map.easeTo({
      center: [point.lng, point.lat],
      zoom: options.zoom ?? 3,
      duration: options.duration ?? 0,
      padding: { ...viewportInsets },
    })
  }

  function setViewportInsets(nextInsets: ViewportInsets): void {
    if (disposed) return
    const next = normalizeViewportInsets(nextInsets)
    if (sameInsets(viewportInsets, next)) return
    viewportInsets = next
    // Do not fit/ease here: this must preserve the user's current camera while
    // moving its visible center into the newly uncovered part of the canvas.
    map.setPadding({ ...viewportInsets })
  }

  function retryStyle(): boolean {
    if (disposed) return false
    map.setStyle(options.styleUrl, { diff: false })
    return true
  }

  return {
    map,
    renderMarkers,
    fitToPoints,
    focusPoint,
    setViewportInsets,
    getViewportInsets: () => ({ ...viewportInsets }),
    markers: () => active,
    retryStyle,
    dispose(): void {
      if (disposed) return
      disposed = true
      for (const marker of active) marker.remove()
      active = []
      map.remove()
    },
  }
}


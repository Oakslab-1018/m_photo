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

export interface MapLifecycleOptions {
  container: HTMLElement
  styleUrl: string
  createMarkerElement: (point: MapPoint) => HTMLElement
  onLoad?: () => void
  onError?: (error: unknown) => void
}

export interface MapLifecycle {
  map: MapLike
  renderMarkers(points: readonly MapPoint[]): void
  fitToPoints(points: readonly MapPoint[], options?: FitToPointsOptions): void
  focusPoint(point: MapPoint, options?: FocusPointOptions): void
  markers(): readonly MarkerLike[]
  dispose(): void
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
  map.on('error', (event) => options.onError?.(event))
  map.on('load', () => options.onLoad?.())

  let active: MarkerLike[] = []
  let disposed = false

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
      map.easeTo({ center: [bounds.west, bounds.south], zoom, duration })
      return
    }
    map.fitBounds(
      [
        [bounds.west, bounds.south],
        [bounds.east, bounds.north],
      ],
      { padding, maxZoom, duration },
    )
  }

  function focusPoint(point: MapPoint, options: FocusPointOptions = {}): void {
    if (disposed) return
    map.easeTo({
      center: [point.lng, point.lat],
      zoom: options.zoom ?? 3,
      duration: options.duration ?? 0,
    })
  }

  return {
    map,
    renderMarkers,
    fitToPoints,
    focusPoint,
    markers: () => active,
    dispose(): void {
      if (disposed) return
      disposed = true
      for (const marker of active) marker.remove()
      active = []
      map.remove()
    },
  }
}


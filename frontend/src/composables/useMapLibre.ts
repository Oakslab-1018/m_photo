import type { LngLatBoundsLike } from '@/utils/geoPoints'

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
  kind: 'country' | 'city'
  iso: string
  name: string
  lat: number
  lng: number
  cosine?: number | null
  scoreType?: string
  confirmed?: boolean
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
  fitToPoints(points: readonly MapPoint[]): void
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

  function fitToPoints(points: readonly MapPoint[]): void {
    if (disposed) return
    const bounds: LngLatBoundsLike | null = boundsFor(points)
    if (!bounds) return
    if (bounds.west === bounds.east && bounds.south === bounds.north) {
      map.easeTo({ center: [bounds.west, bounds.south], zoom: 3, duration: 0 })
      return
    }
    map.fitBounds(
      [
        [bounds.west, bounds.south],
        [bounds.east, bounds.north],
      ],
      { padding: 48, maxZoom: 6, duration: 0 },
    )
  }

  return {
    map,
    renderMarkers,
    fitToPoints,
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

function boundsFor(points: readonly MapPoint[]): LngLatBoundsLike | null {
  if (!points.length) return null
  const lats = points.map((point) => point.lat)
  const lngs = points.map((point) => point.lng).sort((a, b) => a - b)
  const south = Math.min(...lats)
  const north = Math.max(...lats)
  let widest = Number.POSITIVE_INFINITY
  let west = lngs[0]
  let east = lngs[lngs.length - 1]
  for (let index = 0; index < lngs.length; index += 1) {
    const start = lngs[index]
    const end = index === lngs.length - 1 ? lngs[0] + 360 : lngs[index + 1]
    const span = end - start
    if (span < widest) {
      widest = span
      west = start
      east = end
    }
  }
  return { west, south, east, north }
}

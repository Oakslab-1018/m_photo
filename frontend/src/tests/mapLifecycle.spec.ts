import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  createMapLifecycle,
  type MapLibreNamespace,
  type MapLifecycleFailure,
  type MapLike,
  type MapPoint,
  type MarkerLike,
  type ViewportInsets,
} from '@/composables/useMapLibre'

interface FakeMarker extends MarkerLike {
  lngLat: [number, number] | null
  added: boolean
  removed: boolean
}

interface FakeMap extends MapLike {
  options: Record<string, unknown>
  handlers: Record<string, ((event: unknown) => void)[]>
  controls: unknown[]
  fits: Array<{ bounds: unknown; options: unknown }>
  eases: Record<string, unknown>[]
  paddings: ViewportInsets[]
  padding: ViewportInsets
  styleCalls: Array<{ style: string; options?: { diff?: boolean } }>
  center: [number, number]
  zoom: number
  markers: FakeMarker[]
  removed: boolean
  fire(type: string, event?: unknown): void
}

function fakeNamespace(): { api: MapLibreNamespace; last: () => FakeMap } {
  let latest: FakeMap | null = null
  class MapImpl implements FakeMap {
    handlers: Record<string, ((event: unknown) => void)[]> = {}
    controls: unknown[] = []
    fits: Array<{ bounds: unknown; options: unknown }> = []
    eases: Record<string, unknown>[] = []
    paddings: ViewportInsets[] = []
    padding: ViewportInsets = { top: 0, right: 0, bottom: 0, left: 0 }
    styleCalls: Array<{ style: string; options?: { diff?: boolean } }> = []
    center: [number, number] = [0, 20]
    zoom = 1
    markers: FakeMarker[] = []
    removed = false
    constructor(public options: Record<string, unknown>) {
      latest = this
    }
    on(type: string, listener: (event: unknown) => void): this {
      ;(this.handlers[type] ??= []).push(listener)
      return this
    }
    addControl(control: unknown): this {
      this.controls.push(control)
      return this
    }
    fitBounds(bounds: unknown, options: unknown): this {
      this.fits.push({ bounds, options })
      return this
    }
    easeTo(options: Record<string, unknown>): this {
      this.eases.push(options)
      return this
    }
    setPadding(padding: ViewportInsets): this {
      const next = { ...padding }
      this.paddings.push(next)
      this.padding = next
      return this
    }
    setStyle(style: string, options?: { diff?: boolean }): this {
      this.styleCalls.push({ style, options })
      return this
    }
    remove(): void {
      this.removed = true
    }
    fire(type: string, event: unknown = {}): void {
      for (const listener of this.handlers[type] ?? []) listener(event)
    }
  }
  class MarkerImpl implements FakeMarker {
    lngLat: [number, number] | null = null
    added = false
    removed = false
    constructor(private readonly options: { element: HTMLElement }) {}
    setLngLat(lngLat: [number, number]): this {
      this.lngLat = lngLat
      return this
    }
    addTo(map: MapLike): this {
      this.added = true
      const fake = map as FakeMap
      fake.markers.push(this)
      return this
    }
    remove(): void {
      this.removed = true
    }
    element(): HTMLElement {
      return this.options.element
    }
  }
  const api = {
    Map: MapImpl as unknown as MapLibreNamespace['Map'],
    Marker: MarkerImpl as unknown as MapLibreNamespace['Marker'],
    NavigationControl: class {
      constructor(public options?: Record<string, unknown>) {}
    } as unknown as MapLibreNamespace['NavigationControl'],
  }
  return { api, last: () => latest as FakeMap }
}

function point(overrides: Partial<MapPoint> = {}): MapPoint {
  return { key: 'country:JP', kind: 'country', iso: 'JP', name: '日本', lat: 36.2, lng: 138.25, ...overrides }
}

function stubElement(): HTMLElement {
  return {
    classList: { add: () => undefined },
    setAttribute: () => undefined,
    addEventListener: () => undefined,
  } as unknown as HTMLElement
}

function elementFor(_point: MapPoint): HTMLElement {
  return stubElement()
}

function stubContainer(): HTMLElement {
  return {} as HTMLElement
}

describe('createMapLifecycle', () => {
  it('configures demo style, navigation control, and load/error handlers', () => {
    const { api, last } = fakeNamespace()
    let loaded = 0
    let failure: MapLifecycleFailure | undefined
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'https://demotiles.maplibre.org/style.json',
      createMarkerElement: elementFor,
      onLoad: () => (loaded += 1),
      onError: (next) => (failure = next),
    })
    const map = last()
    expect(map.options.style).toBe('https://demotiles.maplibre.org/style.json')
    expect(map.options.trackResize).toBe(true)
    expect(map.controls).toHaveLength(1)
    map.fire('load')
    map.fire('error', { error: new Error('style blocked') })
    expect(loaded).toBe(1)
    expect(failure?.kind).toBe('style')
    expect(failure?.error).toBeInstanceOf(Error)
    lifecycle.dispose()
  })

  it('clears a source warning only after that exact source proves a tile loaded', () => {
    const { api, last } = fakeNamespace()
    const failures: MapLifecycleFailure[] = []
    let recovered = 0
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      onError: (failure) => failures.push(failure),
      onRecover: () => (recovered += 1),
    })
    const map = last()

    map.fire('error', {
      error: new Error('tile blocked'),
      sourceId: 'esri-world-imagery',
      tile: { state: 'errored' },
    })
    expect(failures).toHaveLength(1)
    expect(failures[0]).toMatchObject({
      kind: 'source',
      sourceId: 'esri-world-imagery',
    })

    // load, idle, metadata, and unrelated-source success are not recovery proof.
    map.fire('load')
    map.fire('idle')
    map.fire('sourcedata', {
      sourceId: 'esri-world-imagery',
      sourceDataType: 'metadata',
      isSourceLoaded: true,
    })
    map.fire('sourcedata', {
      sourceId: 'unrelated-source',
      tile: { state: 'loaded' },
    })
    map.fire('sourcedata', {
      sourceId: 'esri-world-imagery',
      sourceDataType: 'idle',
      isSourceLoaded: true,
    })
    expect(recovered).toBe(0)

    map.fire('sourcedata', {
      sourceId: 'esri-world-imagery',
      sourceDataType: 'content',
      tile: { state: 'loaded' },
    })
    expect(recovered).toBe(1)

    map.fire('sourcedata', {
      sourceId: 'esri-world-imagery',
      tile: { state: 'loaded' },
    })
    expect(recovered).toBe(1)
    lifecycle.dispose()
  })

  it('keeps an all-tile outage visible until every failed source has a loaded tile', () => {
    const { api, last } = fakeNamespace()
    let recovered = 0
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      onRecover: () => (recovered += 1),
    })
    const map = last()

    map.fire('error', { sourceId: 'source-a', tile: { state: 'errored' } })
    map.fire('error', { sourceId: 'source-b', tile: { state: 'errored' } })
    map.fire('load')
    map.fire('idle')
    map.fire('style.load')
    map.fire('sourcedata', { sourceId: 'source-a', sourceDataType: 'metadata' })
    map.fire('sourcedata', { sourceId: 'unrelated', tile: { state: 'loaded' } })
    expect(recovered).toBe(0)

    map.fire('sourcedata', { sourceId: 'source-a', tile: { state: 'loaded' } })
    expect(recovered).toBe(0)
    map.fire('sourcedata', { sourceId: 'source-b', tile: { state: 'loaded' } })
    expect(recovered).toBe(1)
    lifecycle.dispose()
  })

  it('does not let source-tile recovery clear a style failure', () => {
    const { api, last } = fakeNamespace()
    let recovered = 0
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      onRecover: () => (recovered += 1),
    })
    const map = last()

    map.fire('error', { sourceId: 'source-a', tile: { state: 'errored' } })
    map.fire('error', { error: new Error('style failed') })
    map.fire('sourcedata', { sourceId: 'source-a', tile: { state: 'loaded' } })
    map.fire('load')
    map.fire('idle')
    expect(recovered).toBe(0)

    map.fire('style.load')
    expect(recovered).toBe(1)
    lifecycle.dispose()
  })

  it('retries the same style with diff disabled while preserving viewport state and markers', () => {
    const { api, last } = fakeNamespace()
    const failures: MapLifecycleFailure[] = []
    let recovered = 0
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo-style-url',
      createMarkerElement: elementFor,
      onError: (failure) => failures.push(failure),
      onRecover: () => (recovered += 1),
    })
    const map = last()

    lifecycle.renderMarkers([point()])
    map.center = [121.5, 31.2]
    map.zoom = 6
    const padding = { top: 12, right: 24, bottom: 36, left: 48 }
    map.setPadding(padding)
    const marker = map.markers[0]
    const fitCount = map.fits.length
    const easeCount = map.eases.length
    const paddingCount = map.paddings.length

    map.fire('error', { sourceId: 'source-a', tile: { state: 'errored' } })
    expect(failures).toHaveLength(1)
    expect(lifecycle.retryStyle()).toBe(true)

    expect(map.styleCalls).toEqual([{ style: 'demo-style-url', options: { diff: false } }])
    expect(map.fits).toHaveLength(fitCount)
    expect(map.eases).toHaveLength(easeCount)
    expect(map.paddings).toHaveLength(paddingCount)
    expect(map.center).toEqual([121.5, 31.2])
    expect(map.zoom).toBe(6)
    expect(map.padding).toEqual(padding)
    expect(map.markers).toHaveLength(1)
    expect(map.markers[0]).toBe(marker)
    expect(marker.removed).toBe(false)

    // Failures from the retried attempt remain visible until their own proof.
    expect(recovered).toBe(0)
    map.fire('error', { sourceId: 'source-a', tile: { state: 'errored' } })
    expect(failures).toHaveLength(2)
    map.fire('sourcedata', { sourceId: 'source-a', tile: { state: 'loaded' } })
    expect(recovered).toBe(1)
    lifecycle.dispose()
  })

  it('replaces markers on every render and removes them on dispose', () => {
    const { api, last } = fakeNamespace()
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
    })
    lifecycle.renderMarkers([point(), point({ key: 'city:jp-tokyo', kind: 'city' })])
    const map = last()
    expect(map.markers).toHaveLength(2)
    const firstBatch = [...map.markers]
    lifecycle.renderMarkers([point({ key: 'country:FR', iso: 'FR', lat: 46.2, lng: 2.2 })])
    expect(firstBatch.every((marker) => marker.removed)).toBe(true)
    expect(map.markers.at(-1)?.lngLat).toEqual([2.2, 46.2])
    lifecycle.dispose()
    expect(map.markers.at(-1)?.removed).toBe(true)
    expect(map.removed).toBe(true)
  })

  it('fits dateline-crossing bounds with an unwrapped east edge', () => {
    const { api, last } = fakeNamespace()
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
    })
    lifecycle.fitToPoints([
      point({ lng: 175 }),
      point({ key: 'country:NZ', lng: -175 }),
    ])
    const fit = last().fits.at(-1)
    expect(fit?.bounds).toEqual([[175, 36.2], [185, 36.2]])
    lifecycle.dispose()
  })

  it('fits multiple points inside asymmetric viewport insets with content margin once', () => {
    const { api, last } = fakeNamespace()
    const viewportInsets = { top: 72, right: 32, bottom: 220, left: 320 }
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      viewportInsets,
    })
    const map = last()

    expect(map.padding).toEqual(viewportInsets)
    lifecycle.fitToPoints(
      [
        point({ key: 'country:JP', lng: 138.25, lat: 36.2 }),
        point({ key: 'country:CN', iso: 'CN', lng: 104.2, lat: 35.8 }),
      ],
      { padding: 36, maxZoom: 4 },
    )

    expect(map.fits.at(-1)?.options).toEqual({
      padding: 36,
      maxZoom: 4,
      duration: 0,
      absolutePadding: false,
    })
    lifecycle.dispose()
  })

  it('honors asymmetric viewport insets for a single fit point and focusPoint', () => {
    const { api, last } = fakeNamespace()
    const viewportInsets = { top: 48, right: 140, bottom: 96, left: 280 }
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      viewportInsets,
    })
    const map = last()
    const japan = point()

    lifecycle.fitToPoints([japan], { singleZoom: 8 })
    expect(map.eases.at(-1)).toMatchObject({
      center: [japan.lng, japan.lat],
      zoom: 8,
      duration: 0,
      padding: viewportInsets,
    })

    lifecycle.focusPoint(point({ key: 'city:tokyo', kind: 'city', lng: 139.69, lat: 35.68 }), {
      zoom: 11,
    })
    expect(map.eases.at(-1)).toMatchObject({
      center: [139.69, 35.68],
      zoom: 11,
      padding: viewportInsets,
    })
    lifecycle.dispose()
  })

  it('updates viewport insets without refitting or moving the current camera', () => {
    const { api, last } = fakeNamespace()
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      viewportInsets: { top: 0, right: 0, bottom: 0, left: 0 },
    })
    const map = last()
    lifecycle.fitToPoints([point(), point({ key: 'country:CN', iso: 'CN', lng: 104.2 })])
    const fitCount = map.fits.length
    const easeCount = map.eases.length
    map.center = [121.5, 31.2]
    map.zoom = 6

    const nextInsets = { top: 64, right: 180, bottom: 260, left: 340 }
    lifecycle.setViewportInsets(nextInsets)

    expect(map.paddings.at(-1)).toEqual(nextInsets)
    expect(map.fits).toHaveLength(fitCount)
    expect(map.eases).toHaveLength(easeCount)
    expect(map.center).toEqual([121.5, 31.2])
    expect(map.zoom).toBe(6)
    expect(lifecycle.getViewportInsets()).toEqual(nextInsets)

    lifecycle.setViewportInsets({ ...nextInsets })
    expect(map.paddings).toHaveLength(2)

    lifecycle.focusPoint(point({ lng: 121.47, lat: 31.23 }), { zoom: 9 })
    expect(map.eases.at(-1)).toMatchObject({ padding: nextInsets })
    lifecycle.dispose()
  })

  it('suppresses callbacks and retries after dispose', () => {
    const { api, last } = fakeNamespace()
    const failures: MapLifecycleFailure[] = []
    let recovered = 0
    let loaded = 0
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
      onError: (failure) => failures.push(failure),
      onRecover: () => (recovered += 1),
      onLoad: () => (loaded += 1),
    })
    const map = last()

    map.fire('error', { sourceId: 'source-a', tile: { state: 'errored' } })
    expect(failures).toHaveLength(1)
    lifecycle.dispose()

    map.fire('sourcedata', { sourceId: 'source-a', tile: { state: 'loaded' } })
    map.fire('style.load')
    map.fire('error', { error: new Error('late style error') })
    map.fire('load')
    expect(failures).toHaveLength(1)
    expect(recovered).toBe(0)
    expect(loaded).toBe(0)
    expect(lifecycle.retryStyle()).toBe(false)
    expect(map.styleCalls).toHaveLength(0)
    expect(map.removed).toBe(true)
    lifecycle.dispose()
  })

  it('ignores renders after dispose and is idempotent', () => {
    const { api, last } = fakeNamespace()
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'demo',
      createMarkerElement: elementFor,
    })
    lifecycle.dispose()
    lifecycle.dispose()
    lifecycle.renderMarkers([point()])
    lifecycle.fitToPoints([point()])
    lifecycle.setViewportInsets({ top: 10, right: 20, bottom: 30, left: 40 })
    const map = last()
    expect(map.markers).toHaveLength(0)
    expect(map.fits).toHaveLength(0)
    expect(map.removed).toBe(true)
  })
})

describe('GeoMap basemap retry UI', () => {
  it('keeps an accessible retry button interactive over the click-through safe area', () => {
    const source = readFileSync(new URL('../components/GeoMap.vue', import.meta.url), 'utf8')

    expect(source).toMatch(/class="ghost-btn map-error-overlay__retry"[\s\S]*?@click\.stop="retryBasemap"/)
    expect(source).toContain('lifecycle?.retryStyle()')
    expect(source).toContain('地图还没准备好，请稍后重试')
    expect(source).toMatch(/\.geo-map__safe-center \.map-error-overlay \{[\s\S]*?pointer-events:\s*auto;/)
  })
})

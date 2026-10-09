import { describe, expect, it } from 'vitest'
import {
  createMapLifecycle,
  type MapLibreNamespace,
  type MapLike,
  type MapPoint,
  type MarkerLike,
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
    let errored = false
    const lifecycle = createMapLifecycle(api, {
      container: stubContainer(),
      styleUrl: 'https://demotiles.maplibre.org/style.json',
      createMarkerElement: elementFor,
      onLoad: () => (loaded += 1),
      onError: () => (errored = true),
    })
    const map = last()
    expect(map.options.style).toBe('https://demotiles.maplibre.org/style.json')
    expect(map.options.trackResize).toBe(true)
    expect(map.controls).toHaveLength(1)
    map.fire('load')
    map.fire('error', { error: new Error('tile blocked') })
    expect(loaded).toBe(1)
    expect(errored).toBe(true)
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
    const map = last()
    expect(map.markers).toHaveLength(0)
    expect(map.fits).toHaveLength(0)
    expect(map.removed).toBe(true)
  })
})

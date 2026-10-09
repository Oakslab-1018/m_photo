/**
 * Map style resolution. No secrets are embedded: VITE_* values are public by
 * definition (Vite inlines them into the deployed bundle), and the MapTiler key
 * is a public client-side key that must be URL-restricted in the MapTiler
 * console. A missing key falls back to the official MapLibre demo tiles, which
 * are demo-only and not for production traffic.
 */
export const MAPLIBRE_DEMO_STYLE_URL = 'https://demotiles.maplibre.org/style.json'

export type MapStyleSource = 'configured' | 'demo'

export interface MapStyleConfig {
  styleUrl: string
  source: MapStyleSource
  /** True when the demo style is used: demo-only, no production SLA. */
  demoOnly: boolean
}

export interface MapStyleEnv {
  VITE_MAP_STYLE_URL?: string
  VITE_MAPTILER_API_KEY?: string
}

function nonEmpty(value: string | undefined): string | null {
  const trimmed = (value ?? '').trim()
  return trimmed ? trimmed : null
}

/** Resolve the effective MapLibre style URL from public build-time config. */
export function resolveMapStyle(env: MapStyleEnv): MapStyleConfig {
  const configured = nonEmpty(env.VITE_MAP_STYLE_URL)
  if (configured) {
    return { styleUrl: configured, source: 'configured', demoOnly: false }
  }
  const key = nonEmpty(env.VITE_MAPTILER_API_KEY)
  if (key) {
    return {
      styleUrl: `https://api.maptiler.com/maps/streets-v4/style.json?key=${encodeURIComponent(key)}`,
      source: 'configured',
      demoOnly: false,
    }
  }
  return { styleUrl: MAPLIBRE_DEMO_STYLE_URL, source: 'demo', demoOnly: true }
}

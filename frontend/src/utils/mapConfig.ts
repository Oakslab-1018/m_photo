/**
 * Map style resolution. No secrets are embedded: VITE_* values are public by
 * definition (Vite inlines them into the deployed bundle), and the MapTiler key
 * is a public client-side key that must be URL-restricted in the MapTiler
 * console. Explicit configuration overrides the bundled Esri World Imagery
 * raster style used by default.
 */
export const ESRI_WORLD_IMAGERY_STYLE_URL = `${import.meta.env.BASE_URL}esri-world-imagery-style.json`

export type MapStyleSource = 'configured' | 'default'

export interface MapStyleConfig {
  styleUrl: string
  source: MapStyleSource
  /** Retained for map UI compatibility; the current default is not demo-only. */
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
  return {
    styleUrl: ESRI_WORLD_IMAGERY_STYLE_URL,
    source: 'default',
    demoOnly: false,
  }
}

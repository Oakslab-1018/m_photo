import type { CountryScore, SourceRegionEntry, SourceRegionRef } from '@/types'

export type { SourceRegionEntry, SourceRegionRef } from '@/types'
/**
 * Shared frontend geopolicy normalization.
 *
 * Upstream model ids may keep raw TW/HK/MO codes for validation only. Every
 * country-level surface in this app renders the canonical CN aggregate; raw
 * regional exclusions stay raw (TW exclusion means 中国台湾地区, not all of CN).
 */
export const CHINA_ISO = 'CN'
export const TAIWAN_ISO = 'TW'
export const HONG_KONG_ISO = 'HK'
export const MACAU_ISO = 'MO'

export const CHINA_REGION_LABELS: Record<string, string> = {
  TW: '中国台湾',
  HK: '中国香港',
  MO: '中国澳门',
}

/** Region scope selector options shown when the country picker is CN. */
export const CHINA_REGION_SCOPES: Array<{ iso: string; label: string }> = [
  { iso: CHINA_ISO, label: '中国大陆' },
  { iso: TAIWAN_ISO, label: '中国台湾' },
  { iso: HONG_KONG_ISO, label: '中国香港' },
  { iso: MACAU_ISO, label: '中国澳门' },
]

/** Canonical China center used only when folding a raw regional row without CN coords. */
export const CHINA_CENTER = { lat: 35.8617, lng: 104.1954 }


export interface GeoRegionMeta {
  iso?: string | null
  name?: string | null
  name_zh?: string | null
  parent_iso?: string | null
  is_region?: boolean | null
  source_regions?: SourceRegionEntry[] | null
  source_iso?: string | null
  region_scope?: string | null
}

export function normalizeIsoCode(iso: string | null | undefined): string {
  return typeof iso === 'string' ? iso.trim().toUpperCase() : ''
}

export function isTaiwanIso(iso: string | null | undefined): boolean {
  return normalizeIsoCode(iso) === TAIWAN_ISO
}

/** True for the raw regional codes that aggregate under country CN. */
export function isChinaRegionCode(iso: string | null | undefined): boolean {
  const code = normalizeIsoCode(iso)
  return code === TAIWAN_ISO || code === HONG_KONG_ISO || code === MACAU_ISO
}

/** Canonical country-level code; raw regional codes fold into CN before display. */
export function normalizeCountryIso(iso: string | null | undefined): string {
  const code = normalizeIsoCode(iso)
  return isChinaRegionCode(code) ? CHINA_ISO : code
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function sourceRegionEntries(row: GeoRegionMeta): SourceRegionEntry[] {
  return Array.isArray(row.source_regions) ? row.source_regions : []
}

export function entryCode(entry: SourceRegionEntry): string {
  if (typeof entry === 'string') return normalizeIsoCode(entry)
  const record = isRecord(entry) ? entry : {}
  return normalizeIsoCode(
    (typeof record.source_iso === 'string' ? record.source_iso : null) ??
      (typeof record.iso === 'string' ? record.iso : null),
  )
}

function entryLabel(entry: SourceRegionEntry): string | null {
  if (typeof entry === 'string') return CHINA_REGION_LABELS[normalizeIsoCode(entry)] ?? null
  const record = isRecord(entry) ? entry : {}
  const explicit = typeof record.name_zh === 'string' ? record.name_zh.trim() : ''
  if (explicit) return explicit
  return CHINA_REGION_LABELS[entryCode(entry)] ?? null
}

function sourceRegionCodes(row: GeoRegionMeta): string[] {
  return sourceRegionEntries(row).map(entryCode).filter(Boolean)
}

/** True when a country-level row is (or aggregates) a CN region such as TW/HK/MO. */
export function isChinaRegionRow(row: GeoRegionMeta | null | undefined): boolean {
  if (!row) return false
  if (isChinaRegionCode(row.iso)) return true
  const rawName = `${row.name_zh ?? ''} ${row.name ?? ''}`.trim()
  if (rawName && /^(台湾|台灣|taiwan)$/i.test(rawName)) return true
  return sourceRegionCodes(row).some(isChinaRegionCode)
}

/** Back-compat helper: a row that specifically carries raw Taiwan evidence. */
export function isTaiwanRow(row: GeoRegionMeta | null | undefined): boolean {
  if (!row) return false
  if (isTaiwanIso(row.iso)) return true
  const rawName = `${row.name_zh ?? ''} ${row.name ?? ''}`.trim()
  if (rawName && /^(台湾|台灣|taiwan)$/i.test(rawName)) return true
  return sourceRegionCodes(row).includes(TAIWAN_ISO)
}

/**
 * Optional region-scope label rendered next to the canonical CN row, e.g.
 * 中国台湾. Prefers explicit backend metadata, then source-region names, then
 * the raw regional iso itself.
 */
export function regionScopeLabel(row: GeoRegionMeta | null | undefined): string | null {
  if (!row) return null
  const explicit = typeof row.region_scope === 'string' ? row.region_scope.trim() : ''
  if (explicit) return explicit
  const labels = sourceRegionEntries(row).map(entryLabel).filter((label): label is string => Boolean(label))
  const own = isChinaRegionCode(row.iso) ? CHINA_REGION_LABELS[normalizeIsoCode(row.iso)] : null
  if (own) labels.unshift(own)
  return labels.length ? [...new Set(labels)].join('、') : null
}

/** Catalogue-only label: raw regional codes stay addressable as region scopes. */
export function catalogCountryLabel(row: GeoRegionMeta | null | undefined): string {
  if (!row) return ''
  const own = isChinaRegionCode(row.iso) ? CHINA_REGION_LABELS[normalizeIsoCode(row.iso)] : null
  if (own) return own
  return (row.name_zh || row.name || normalizeIsoCode(row.iso)).trim()
}

/** Country-level display name; any CN region row renders as 中国. */
export function displayCountryName(row: GeoRegionMeta | null | undefined): string {
  if (!row) return ''
  if (isChinaRegionRow(row)) return '中国'
  return (row.name_zh || row.name || normalizeIsoCode(row.iso)).trim()
}

/**
 * Exclusion labels stay region-accurate: raw TW/HK/MO mean only that region is
 * excluded, so they must never be canonicalized into 中国 on the client.
 */
export function exclusionLabel(iso: string | null | undefined): string {
  const code = normalizeIsoCode(iso)
  if (code === CHINA_ISO) return '中国'
  const region = CHINA_REGION_LABELS[code]
  return region ? `${region}地区` : code
}

/**
 * Region label for one city row: `iso` may already be the canonical CN while
 * `source_iso`/`source_region` carry the raw TW/HK/MO region. Taiwan cities
 * keep their own name; only their region parentage is labelled.
 */
export function cityRegionLabel(
  city: { iso?: string | null; source_iso?: string | null; source_region?: SourceRegionEntry | null } | null | undefined,
): string | null {
  if (!city) return null
  const regionCode = city.source_region ? entryCode(city.source_region) : null
  for (const code of [city.source_iso, regionCode, city.iso]) {
    const label = CHINA_REGION_LABELS[normalizeIsoCode(code)]
    if (label) return label
  }
  const raw = typeof city.source_region === 'string' ? city.source_region.trim() : ''
  return raw || (city.source_region ? entryLabel(city.source_region) : null)
}

function mergeSourceRegionEntries(
  ...groups: Array<SourceRegionEntry[] | null | undefined>
): SourceRegionEntry[] {
  const seen = new Set<string>()
  const out: SourceRegionEntry[] = []
  for (const group of groups) {
    for (const entry of group ?? []) {
      const code = entryCode(entry)
      const label = entryLabel(entry)
      const key = `${code}\u0000${label ?? ''}`
      if (!code && !label) continue
      if (seen.has(key)) continue
      seen.add(key)
      out.push(entry)
    }
  }
  return out
}

function regionFallbackEntry(code: string): SourceRegionRef {
  return {
    source_iso: code,
    name_zh: CHINA_REGION_LABELS[code] ?? null,
    parent_iso: CHINA_ISO,
  }
}

/** Fold one raw score row into the canonical app-level country row. */
export function normalizeCountryScore(score: CountryScore): CountryScore {
  const rawIso = normalizeIsoCode(score.iso)
  const regionRow = isChinaRegionRow(score)
  const iso = regionRow ? CHINA_ISO : rawIso
  if (!regionRow && iso === rawIso) return score
  const regionalFallback = regionRow && rawIso !== CHINA_ISO
  return {
    ...score,
    iso,
    name: regionRow ? 'China' : score.name,
    name_zh: regionRow ? '中国' : score.name_zh,
    // CN country coordinates are preserved; a raw regional row without a CN
    // sibling falls back to the canonical China center instead of Taipei.
    lat: regionalFallback ? CHINA_CENTER.lat : score.lat,
    lng: regionalFallback ? CHINA_CENTER.lng : score.lng,
    parent_iso: regionRow ? score.parent_iso ?? CHINA_ISO : score.parent_iso,
    is_region: false,
    source_regions: mergeSourceRegionEntries(
      score.source_regions,
      regionalFallback ? [regionFallbackEntry(rawIso)] : null,
    ),
    region_scope: regionScopeLabel(score),
  }
}

function scoreValue(score: CountryScore): number {
  return Number.isFinite(score.score) ? score.score : 0
}

/**
 * Normalize and dedupe country scores so raw TW/HK/MO rows can never render as
 * separate country entries next to 中国(CN). When both a CN row and a regional
 * row exist, the CN row keeps its identity and coordinates while the regional
 * evidence is merged into source_regions.
 */
export function normalizeCountryScores(
  scores: readonly CountryScore[] | null | undefined,
): CountryScore[] {
  const grouped = new Map<string, CountryScore[]>()
  for (const raw of scores ?? []) {
    if (!raw || typeof raw !== 'object') continue
    const key = normalizeCountryIso(raw.iso) || '_'
    const list = grouped.get(key) ?? []
    list.push(raw)
    grouped.set(key, list)
  }
  const out: CountryScore[] = []
  for (const rows of grouped.values()) {
    if (rows.length === 1) {
      out.push(normalizeCountryScore(rows[0]))
      continue
    }
    const mainland = rows.find((row) => normalizeIsoCode(row.iso) === CHINA_ISO)
    const base = mainland ?? rows.reduce((best, row) => (scoreValue(row) > scoreValue(best) ? row : best), rows[0])
    const others = rows.filter((row) => row !== base)
    const mergedSourceRegions = mergeSourceRegionEntries(
      base.source_regions,
      ...others.map((row) => row.source_regions),
      ...others
        .map((row) => normalizeIsoCode(row.iso))
        .filter((code) => isChinaRegionCode(code))
        .map((code) => [regionFallbackEntry(code)]),
    )
    out.push(
      normalizeCountryScore({
        ...base,
        score: Math.max(...rows.map(scoreValue)),
        source_regions: mergedSourceRegions,
      }),
    )
  }
  return out
}
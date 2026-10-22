<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type {
  CityScore,
  CountryScore,
  GuidedStage,
  LocationCandidate,
  LocationTarget,
} from '@/types'
import { cityMatchedClueCount, cityScoreLabel, displayCityName } from '@/utils/city'
import {
  candidateScoreLabel,
  guidedCandidateComposition,
  locationLevelLabel,
  selectTopGuidedCandidates,
  targetLabel,
} from '@/utils/guided'
import { scoreLabel } from '@/utils/map'
import {
  CHINA_REGION_LABELS,
  CHINA_REGION_SCOPES,
  cityRegionLabel,
  displayCountryName,
  normalizeCountryIso,
  normalizeIsoCode,
  regionScopeLabel,
} from '@/utils/geo'
import ReferenceMatches from '@/components/ReferenceMatches.vue'
import PlaceMatchEvidence from '@/components/PlaceMatchEvidence.vue'

const props = defineProps<{
  scores: CountryScore[]
  confirmedIso: string | null
  selectedIso: string | null
  /** Region-aware exclusion chips: raw TW/HK/MO codes keep their own label. */
  exclusions: Array<{ iso: string; label: string }>
  busy: boolean
  cityScores?: CityScore[]
  cityCountryIso?: string | null
  citiesStatus?: string | null
  countryOptions: Array<{ iso: string; name_zh: string }>
  cityBusy: boolean
  cityNotice: string | null
  /** Requested CN region filter ('' = merged all regions). */
  citySourceIso?: string | null
  /** Response-provided region label when the request was filtered. */
  citySourceLabel?: string | null
  embedded?: boolean
  collapsible?: boolean
  collapsed?: boolean
  /** Guided flow is active: this panel is evidence-only for lower stages. */
  guidedMode?: boolean
  /** Country actions can be sent only while the guided stage is country. */
  guidedCountryAllowed?: boolean
  /** Guided server candidates, already limited to the current stage. */
  guidedCandidates?: LocationCandidate[]
  /** False while pending observations still hide the country/region rankings. */
  guidedCandidatesUnlocked?: boolean
  /** False when a newly appended observation still needs review. */
  guidedConfirmationReady?: boolean
  guidedDataReady?: boolean
  /** Current server-guided level used to keep the candidate tab label accurate. */
  guidedStage?: GuidedStage | null
  /** Render candidates, references, or both. The map never lives in this panel. */
  surface?: 'all' | 'candidates' | 'references'
}>()

const emit = defineEmits<{
  confirm: [score: CountryScore]
  reject: [score: CountryScore]
  'select-country': [iso: string]
  'select-guided-target': [target: LocationTarget]
  'confirm-guided-target': [target: LocationTarget]
  'reject-guided-target': [target: LocationTarget]
  'request-cities': [iso: string, sourceIso: string | null]
  'toggle-collapse': []
  expand: []
}>()

type CandidateTab = 'country' | 'city' | 'reference'

const ALL_TABS: Array<{ key: CandidateTab; label: string }> = [
  { key: 'country', label: '国家' },
  { key: 'city', label: '城市' },
  { key: 'reference', label: '参考' },
]

const guidedTabLabel = computed(() => {
  switch (props.guidedStage) {
    case 'country':
      return '国家 / 地区'
    case 'city':
      return '城市候选'
    default:
      return '地点候选'
  }
})
const tabs = computed(() => {
  const surface = props.surface ?? 'all'
  if (surface === 'references') return []
  let available = surface === 'candidates'
    ? ALL_TABS.filter((tab) => tab.key !== 'reference')
    : [...ALL_TABS]
  // In guided mode the server-ranked Top-5 is the only candidate source; the
  // legacy directory city tab would bypass that contract.
  if (props.guidedMode) available = available.filter((tab) => tab.key !== 'city')
  return available.map((tab) =>
    tab.key === 'country' && props.guidedMode
      ? { ...tab, label: guidedTabLabel.value }
      : tab,
  )
})
const initialTab = (): CandidateTab => props.surface === 'references' ? 'reference' : 'country'
const activeTab = ref<CandidateTab>(initialTab())

const guidedTopCandidates = computed(() =>
  props.guidedMode
    ? selectTopGuidedCandidates(props.guidedCandidates ?? [], props.guidedStage)
    : [],
)
const guidedComposition = computed(() => guidedCandidateComposition(guidedTopCandidates.value))
const guidedCandidateLevelName = computed(() =>
  props.guidedStage === 'city' ? '城市' : '国家 / 地区',
)
const guidedSectionNote = computed(() => {
  const composition = guidedComposition.value
  if (composition.hasUserText) {
    return composition.mixed ? '含线索匹配' : '线索匹配候选'
  }
  if (composition.hasCatalogSearch) {
    return composition.mixed ? '含目录检索' : '目录检索候选'
  }
  if (composition.hasModelScores) return '候选前五'
  return `目录候选（${guidedTopCandidates.value.length} 项）`
})
const guidedSectionHint = computed(() => {
  const composition = guidedComposition.value
  if (!props.guidedConfirmationReady) {
    return '有新的观察待核对，暂时不能确认或排除。'
  }
  if (composition.hasUserText) {
    return '标有“线索匹配候选”的条目来自你输入的文字。'
  }
  if (composition.hasCatalogSearch) {
    return '目录检索命中的条目优先显示。'
  }
  if (composition.hasModelScores) {
    return '匹配度仅供参考。'
  }
  return `按目录顺序显示；可输入${guidedCandidateLevelName.value}名称，或补充照片里的线索。`
})
const countryCount = computed(() =>
  props.guidedMode
    ? props.guidedCandidatesUnlocked
      ? guidedTopCandidates.value.length
      : 0
    : props.scores.length,
)

watch(tabs, (current) => {
  if (!current.length) {
    activeTab.value = 'reference'
    return
  }
  if (!current.some((tab) => tab.key === activeTab.value)) activeTab.value = 'country'
})

watch(
  () => props.surface,
  (surface) => {
    activeTab.value = surface === 'references' ? 'reference' : 'country'
  },
)

const cityPickerIso = ref('')
const citySource = ref('')

const isChinaSelection = computed(() => normalizeIsoCode(cityPickerIso.value) === 'CN')
const excludedRegionCodes = computed(
  () => new Set(props.exclusions.map((item) => normalizeIsoCode(item.iso))),
)
const visibleRegionScopes = computed(() =>
  CHINA_REGION_SCOPES.filter((scope) => !excludedRegionCodes.value.has(scope.iso)),
)

watch(
  () => [props.confirmedIso, props.cityCountryIso, props.scores.length] as const,
  () => {
    cityPickerIso.value =
      props.confirmedIso ?? props.cityCountryIso ?? props.scores[0]?.iso ?? ''
    if (!isChinaSelection.value) citySource.value = ''
  },
  { immediate: true },
)

watch(
  () => props.citySourceIso,
  (value) => {
    citySource.value = normalizeIsoCode(value ?? '')
  },
)

const referenceCount = computed(() => {
  const ids = new Set<string>()
  for (const score of props.scores) {
    for (const match of score.reference_matches ?? []) {
      if (match?.reference_id) ids.add(match.reference_id)
    }
  }
  return ids.size
})

const cityCount = computed(() => props.cityScores?.length ?? 0)

const cityScopeName = computed(() => {
  const iso = normalizeCountryIso(props.cityCountryIso)
  return props.countryOptions.find((option) => option.iso === iso)?.name_zh ?? iso
})

const cityFilterLabel = computed(() => {
  if (props.citySourceLabel) return props.citySourceLabel
  const source = normalizeIsoCode(props.citySourceIso ?? '')
  return source ? CHINA_REGION_LABELS[source] ?? source : ''
})

function selectTab(key: CandidateTab): void {
  activeTab.value = key
  if (props.collapsed) emit('expand')
}

function requestCities(): void {
  if (!cityPickerIso.value) return
  emit('request-cities', cityPickerIso.value, citySource.value || null)
}

function onCountryChange(): void {
  citySource.value = ''
  requestCities()
}

function onScopeChange(): void {
  requestCities()
}

function isForeignCitySelection(): boolean {
  return Boolean(
    props.cityCountryIso &&
      cityPickerIso.value &&
      normalizeCountryIso(props.cityCountryIso) !== normalizeCountryIso(cityPickerIso.value),
  )
}
</script>

<template>
  <section
    class="panel candidate-panel"
    :class="[embedded ? 'embedded' : '', `candidate-panel--${surface ?? 'all'}`]"
  >
    <div v-if="tabs.length || collapsible" class="candidate-tabs" role="tablist" aria-label="候选结果">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        role="tab"
        :aria-selected="activeTab === tab.key"
        :class="{ active: activeTab === tab.key }"
        @click="selectTab(tab.key)"
      >
        {{ tab.label }}
        <span v-if="tab.key === 'country' && countryCount" class="tab-count">{{ countryCount }}</span>
        <span v-else-if="tab.key === 'city' && cityCount" class="tab-count">{{ cityCount }}</span>
        <span v-else-if="tab.key === 'reference' && referenceCount" class="tab-count">{{ referenceCount }}</span>
      </button>
      <button
        v-if="collapsible"
        type="button"
        class="tabs-toggle"
        :aria-expanded="!collapsed"
        :aria-label="collapsed ? '展开候选面板' : '收起候选面板'"
        :title="collapsed ? '展开候选面板' : '收起候选面板'"
        @click="emit('toggle-collapse')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" :class="{ open: !collapsed }">
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
    </div>

    <div v-if="surface !== 'references'" v-show="activeTab === 'country' && !collapsed" class="candidate-body">
      <template v-if="guidedMode">
        <div v-if="guidedDataReady === false" class="candidate-empty">
          正在加载…
          <span class="next">加载完成前不会显示旧候选</span>
        </div>
        <div v-else-if="!guidedCandidatesUnlocked" class="candidate-empty candidate-locked">
          先核对观察，再查看候选
          <span class="next">核对完成后，这里会显示候选。</span>
        </div>
        <template v-else>
          <div v-if="!guidedTopCandidates.length" class="candidate-empty">
            暂无候选
            <span class="next">可在下方补充线索，或直接输入地名。</span>
          </div>
          <template v-else>
            <div class="panel-hint">{{ guidedSectionNote }}</div>
            <div class="panel-hint">{{ guidedSectionHint }}</div>
            <article
              v-for="(candidate, index) in guidedTopCandidates"
              :key="`${candidate.target.level}:${candidate.target.id}`"
              class="candidate-card"
              role="button"
              tabindex="0"
              @click="emit('select-guided-target', candidate.target)"
              @keydown.enter.prevent="emit('select-guided-target', candidate.target)"
              @keydown.space.prevent="emit('select-guided-target', candidate.target)"
            >
              <div class="candidate-main">
                <span class="candidate-rank">{{ index + 1 }}</span>
                <div>
                  <div class="candidate-name">
                    {{ targetLabel(candidate.target) }}
                  </div>
                  <div class="candidate-score">{{ candidateScoreLabel(candidate) }}</div>
                  <PlaceMatchEvidence v-if="candidate.place_match" :match="candidate.place_match" />
                </div>
              </div>
              <div class="candidate-actions">
                <button
                  type="button"
                  class="mini-btn confirm"
                  :disabled="busy || !guidedConfirmationReady"
                  @click.stop="emit('confirm-guided-target', candidate.target)"
                >确认{{ locationLevelLabel(candidate.target.level) }}</button>
                <button
                  type="button"
                  class="mini-btn reject"
                  :disabled="busy || !guidedConfirmationReady"
                  @click.stop="emit('reject-guided-target', candidate.target)"
                >排除{{ locationLevelLabel(candidate.target.level) }}</button>
              </div>
            </article>
          </template>
        </template>
      </template>

      <template v-else>
        <div v-if="!scores.length" class="candidate-empty">
          暂无候选国家
          <span class="next">先上传照片并完成分析</span>
        </div>
        <template v-else>
          <div class="panel-hint">
            按顺序排列 · 地图位置不是照片 GPS · 确认前会再次询问
          </div>
          <article
            v-for="(candidate, index) in scores"
            :key="candidate.iso"
            class="candidate-card"
            :class="{ selected: selectedIso === candidate.iso, confirmed: confirmedIso === candidate.iso }"
            role="button"
            tabindex="0"
            :aria-pressed="selectedIso === candidate.iso"
            @click="emit('select-country', candidate.iso)"
            @keydown.enter.prevent="emit('select-country', candidate.iso)"
            @keydown.space.prevent="emit('select-country', candidate.iso)"
          >
            <div class="candidate-main">
              <span class="candidate-rank">{{ index + 1 }}</span>
              <div>
                <div class="candidate-name">
                  {{ displayCountryName(candidate) }}
                  <span v-if="regionScopeLabel(candidate)" class="tag region">
                    地区 · {{ regionScopeLabel(candidate) }}
                  </span>
                  <span v-if="confirmedIso === candidate.iso" class="tag">已确认</span>
                </div>
                <div class="candidate-score">{{ scoreLabel(candidate) }}</div>
              </div>
            </div>
            <div class="candidate-actions">
              <button
                type="button"
                class="mini-btn confirm"
                :disabled="busy"
                @click.stop="emit('confirm', candidate)"
              >确认</button>
              <button
                type="button"
                class="mini-btn reject"
                :disabled="busy"
                @click.stop="emit('reject', candidate)"
              >排除</button>
            </div>
          </article>
          <div v-if="exclusions.length" class="exclusion-note">
            <span class="exclusion-prefix">已排除</span>
            <span v-for="item in exclusions" :key="item.iso" class="exclusion-chip">{{ item.label }}</span>
            <span class="exclusion-hint">只影响该地区</span>
          </div>
        </template>
      </template>
    </div>

    <div v-if="surface !== 'references' && !guidedMode" v-show="activeTab === 'city' && !collapsed" class="candidate-body">
      <div v-if="guidedMode && guidedDataReady === false" class="candidate-empty">
        正在加载…
        <span class="next">加载完成前不会显示旧的城市候选</span>
      </div>
      <div v-else-if="guidedMode && !guidedCandidatesUnlocked" class="candidate-empty candidate-locked">
        先核对观察，再查看城市候选
        <span class="next">核对完成后会显示城市候选。</span>
      </div>
      <template v-else>
      <div class="city-toolbar">
        <label>
          选择国家/地区
          <select v-model="cityPickerIso" :disabled="busy || cityBusy" @change="onCountryChange">
            <option value="" disabled>— 选择国家/地区 —</option>
            <option v-for="option in countryOptions" :key="option.iso" :value="option.iso">
              {{ option.name_zh }}（{{ option.iso }}）
            </option>
          </select>
        </label>
        <label v-if="isChinaSelection">
          区域范围
          <select v-model="citySource" :disabled="busy || cityBusy" @change="onScopeChange">
            <option value="">全部地区</option>
            <option v-for="scope in visibleRegionScopes" :key="scope.iso" :value="scope.iso">
              {{ scope.label }}
            </option>
          </select>
        </label>
        <div class="panel-hint">
          {{ confirmedIso ? '已选国家的城市优先；切换国家只改变查询，不改变确认' : '只用于查询城市，不会确认国家' }}
        </div>
        <button type="button" class="ghost-btn" :disabled="!cityPickerIso || cityBusy || busy" @click="requestCities">
          {{ cityBusy ? '查询中…' : '查询城市' }}
        </button>
      </div>
      <div v-if="isForeignCitySelection()" class="inline-note">
        正在查看「{{ cityPickerIso }}」的城市，之前的结果可能属于别的国家。
      </div>

      <div v-if="cityNotice" class="inline-note">{{ cityNotice }}</div>
      <div v-if="!cityCount" class="candidate-empty">
        暂无城市候选
        <span class="next">选择国家/地区后查询；城市位置是代表点，不是照片位置</span>
      </div>
      <template v-else>
        <div class="panel-hint">
          {{ cityScopeName }}<template v-if="cityFilterLabel"> · {{ cityFilterLabel }}</template>
          · 共 {{ cityCount }} 个 · 位置不是照片 GPS
        </div>
        <article v-for="city in cityScores" :key="city.city_id" class="candidate-card city-card">
          <div class="candidate-main">
            <span class="candidate-rank city">城</span>
            <div>
              <div class="candidate-name">
                {{ displayCityName(city) }}
                <span v-if="cityRegionLabel(city)" class="tag region">{{ cityRegionLabel(city) }}</span>
              </div>
              <div class="candidate-score">{{ city.region }} · {{ cityScoreLabel(city) }}</div>
              <div v-if="cityMatchedClueCount(city)" class="candidate-score city-clues">
                匹配 {{ cityMatchedClueCount(city) }} 条线索
              </div>
            </div>
          </div>
        </article>
      </template>
      </template>
    </div>

    <div v-if="surface !== 'candidates'" v-show="activeTab === 'reference' && !collapsed" class="candidate-body reference-tab">
      <div v-if="guidedMode && guidedDataReady === false" class="candidate-empty">
        正在加载…
      </div>
      <div v-else-if="guidedMode && !guidedCandidatesUnlocked" class="candidate-empty candidate-locked">
        先核对观察，再查看参考街景
        <span class="next">核对完成前，参考候选不会提前显示。</span>
      </div>
      <ReferenceMatches v-else :scores="scores" />
    </div>
  </section>
</template>

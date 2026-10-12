<script setup lang="ts">
import { computed, ref, useSlots, watch } from 'vue'
import type { CityScore, CountryScore } from '@/types'
import { cityMatchedClueCount, cityScoreLabel } from '@/utils/city'
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
}>()

const emit = defineEmits<{
  confirm: [score: CountryScore]
  reject: [score: CountryScore]
  'select-country': [iso: string]
  'request-cities': [iso: string, sourceIso: string | null]
  'toggle-collapse': []
  expand: []
}>()

type CandidateTab = 'map' | 'country' | 'city' | 'reference'

const ALL_TABS: Array<{ key: CandidateTab; label: string }> = [
  { key: 'map', label: '地图' },
  { key: 'country', label: '国家' },
  { key: 'city', label: '城市' },
  { key: 'reference', label: '参考' },
]

const slots = useSlots()
const tabs = computed(() => (slots.map ? ALL_TABS : ALL_TABS.filter((tab) => tab.key !== 'map')))
const activeTab = ref<CandidateTab>('map')

watch(tabs, (current) => {
  if (!current.some((tab) => tab.key === activeTab.value)) activeTab.value = 'country'
})

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
  <section class="panel candidate-panel" :class="{ embedded }">
    <div class="candidate-tabs" role="tablist" aria-label="候选结果">
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
        <span v-if="tab.key === 'country'" class="tab-count">{{ scores.length }}</span>
        <span v-else-if="tab.key === 'city' && cityCount" class="tab-count">{{ cityCount }}</span>
        <span v-else-if="tab.key === 'reference' && referenceCount" class="tab-count">{{ referenceCount }}</span>
      </button>
      <button
        v-if="collapsible"
        type="button"
        class="tabs-toggle"
        :aria-expanded="!collapsed"
        :aria-label="collapsed ? '展开推理面板' : '收起推理面板'"
        :title="collapsed ? '展开推理面板' : '收起推理面板'"
        @click="emit('toggle-collapse')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" :class="{ open: !collapsed }">
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
    </div>

    <div v-show="activeTab === 'map' && !collapsed" class="candidate-body map-tab">
      <slot name="map" />
    </div>

    <div v-show="activeTab === 'country' && !collapsed" class="candidate-body">
      <div v-if="!scores.length" class="candidate-empty">
        暂无候选国家
        <span class="next">先上传照片并完成推理</span>
      </div>
      <template v-else>
        <div class="panel-hint">
          {{
            guidedMode
              ? (guidedCountryAllowed
                ? '引导模式：这里的确认 / 排除会提交引导协议事件'
                : '引导进行中：国家候选仅供浏览，请在主引导面板完成当前步骤')
              : '按序候选 · 代表点不是照片 GPS；确认需二次确认'
          }}
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
                  区域命中 · {{ regionScopeLabel(candidate) }}
                </span>
                <span v-if="confirmedIso === candidate.iso" class="tag">已确认国家</span>
              </div>
              <div class="candidate-score">{{ scoreLabel(candidate) }}</div>
            </div>
          </div>
          <div class="candidate-actions">
            <button
              type="button"
              class="mini-btn confirm"
              :disabled="busy || (guidedMode && !guidedCountryAllowed)"
              @click.stop="emit('confirm', candidate)"
            >确认</button>
            <button
              type="button"
              class="mini-btn reject"
              :disabled="busy || (guidedMode && !guidedCountryAllowed)"
              @click.stop="emit('reject', candidate)"
            >排除</button>
          </div>
        </article>
        <div v-if="exclusions.length" class="exclusion-note">
          <span class="exclusion-prefix">已排除</span>
          <span v-for="item in exclusions" :key="item.iso" class="exclusion-chip">{{ item.label }}</span>
          <span class="exclusion-hint">区域排除仅影响该地区</span>
        </div>
      </template>
    </div>

    <div v-show="activeTab === 'city' && !collapsed" class="candidate-body">
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
          {{ confirmedIso ? '优先已确认国家，可浏览其他国家/地区（浏览不改变确认）' : '国家/地区选择仅为查询城市候选，不代表确认' }}
        </div>
        <button type="button" class="ghost-btn" :disabled="!cityPickerIso || cityBusy || busy" @click="requestCities">
          {{ cityBusy ? '查询中…' : '查询城市候选' }}
        </button>
      </div>
      <div v-if="isForeignCitySelection()" class="inline-note">
        正在查看「{{ cityPickerIso }}」的城市候选，此前结果可能属于其他国家/地区。
      </div>

      <div v-if="cityNotice" class="inline-note">{{ cityNotice }}</div>
      <div v-if="!cityCount" class="candidate-empty">
        暂无城市候选
        <span class="next">选择国家/地区后查询；城市仅为代表点，不是照片位置</span>
      </div>
      <template v-else>
        <div class="panel-hint">
          {{ cityScopeName }}<template v-if="cityFilterLabel"> · {{ cityFilterLabel }}</template>
          城市候选代表点 · 非照片定位 · 共 {{ cityCount }} 个
        </div>
        <article v-for="city in cityScores" :key="city.city_id" class="candidate-card city-card">
          <div class="candidate-main">
            <span class="candidate-rank city">城</span>
            <div>
              <div class="candidate-name">
                {{ city.name }}
                <span v-if="cityRegionLabel(city)" class="tag region">{{ cityRegionLabel(city) }}</span>
              </div>
              <div class="candidate-score">{{ city.region }} · {{ cityScoreLabel(city) }}</div>
              <div v-if="cityMatchedClueCount(city)" class="candidate-score city-clues">
                命中线索 {{ cityMatchedClueCount(city) }} 条（排序诊断，非概率）
              </div>
            </div>
          </div>
        </article>
      </template>
    </div>

    <div v-show="activeTab === 'reference' && !collapsed" class="candidate-body reference-tab">
      <ReferenceMatches :scores="scores" />
    </div>
  </section>
</template>

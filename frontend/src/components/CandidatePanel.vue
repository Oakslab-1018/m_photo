<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CityScore, CountryScore } from '@/types'
import { cityMatchedClueCount, cityScoreLabel } from '@/utils/city'
import { scoreLabel } from '@/utils/map'
import ReferenceMatches from '@/components/ReferenceMatches.vue'

const props = defineProps<{
  scores: CountryScore[]
  confirmedIso: string | null
  selectedIso: string | null
  exclusions: string[]
  busy: boolean
  cityScores?: CityScore[]
  cityCountryIso?: string | null
  citiesStatus?: string | null
  countryOptions: Array<{ iso: string; name_zh: string }>
  cityBusy: boolean
  cityNotice: string | null
}>()

const emit = defineEmits<{
  confirm: [score: CountryScore]
  reject: [score: CountryScore]
  'select-country': [iso: string]
  'request-cities': [iso: string]
}>()

const tabs = [
  { key: 'country', label: '国家' },
  { key: 'city', label: '城市' },
  { key: 'reference', label: '参考' },
] as const

const activeTab = ref<(typeof tabs)[number]['key']>('country')
const cityPickerIso = ref('')

watch(
  () => [props.confirmedIso, props.cityCountryIso, props.scores.length] as const,
  () => {
    cityPickerIso.value =
      props.confirmedIso ?? props.cityCountryIso ?? props.scores[0]?.iso ?? ''
  },
  { immediate: true },
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

function requestCities(): void {
  if (!cityPickerIso.value) return
  emit('request-cities', cityPickerIso.value)
}

function isForeignCitySelection(): boolean {
  return Boolean(
    props.cityCountryIso &&
      cityPickerIso.value &&
      props.cityCountryIso !== cityPickerIso.value,
  )
}


</script>

<template>
  <section class="panel candidate-panel">
    <div class="candidate-tabs" role="tablist" aria-label="候选结果">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        role="tab"
        :aria-selected="activeTab === tab.key"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
        <span v-if="tab.key === 'country'" class="tab-count">{{ scores.length }}</span>
        <span v-else-if="tab.key === 'city' && cityCount" class="tab-count">{{ cityCount }}</span>
        <span v-else-if="tab.key === 'reference' && referenceCount" class="tab-count">{{ referenceCount }}</span>
      </button>
    </div>

    <div v-if="activeTab === 'country'" class="candidate-body">
      <div v-if="!scores.length" class="candidate-empty">
        暂无候选国家
        <span class="next">先上传照片并完成推理</span>
      </div>
      <template v-else>
        <div class="panel-hint">按序候选 · 代表点不是照片 GPS；确认需二次确认</div>
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
                {{ candidate.name_zh }}
                <span v-if="confirmedIso === candidate.iso" class="tag">已确认国家</span>
              </div>
              <div class="candidate-score">{{ scoreLabel(candidate) }}</div>
            </div>
          </div>
          <div class="candidate-actions">
            <button type="button" class="mini-btn confirm" :disabled="busy" @click.stop="emit('confirm', candidate)">确认</button>
            <button type="button" class="mini-btn reject" :disabled="busy" @click.stop="emit('reject', candidate)">排除</button>
          </div>
        </article>
        <div v-if="exclusions.length" class="exclusion-note">已排除：{{ exclusions.join('、') }}</div>
      </template>
    </div>

    <div v-else-if="activeTab === 'city'" class="candidate-body">
      <div class="city-toolbar">
        <label>
          选择国家
          <select v-model="cityPickerIso" :disabled="busy || cityBusy" @change="requestCities">
            <option value="" disabled>— 选择国家目录 —</option>
            <option v-for="option in countryOptions" :key="option.iso" :value="option.iso">
              {{ option.name_zh }}（{{ option.iso }}）
            </option>
          </select>
        </label>
        <div class="panel-hint">
          {{ confirmedIso ? '优先已确认国家，可浏览其他国家（浏览不改变确认）' : '国家选择仅为查询城市候选，不代表确认' }}
        </div>
        <button type="button" class="ghost-btn" :disabled="!cityPickerIso || cityBusy || busy" @click="requestCities">
          {{ cityBusy ? '查询中…' : '查询城市候选' }}
        </button>
      </div>
      <div v-if="isForeignCitySelection()" class="inline-note">
        正在查看「{{ cityPickerIso }}」的城市候选，此前结果可能属于其他国家。
      </div>

      <div v-if="cityNotice" class="inline-note">{{ cityNotice }}</div>
      <div v-if="!cityCount" class="candidate-empty">
        暂无城市候选
        <span class="next">选择国家后查询；城市仅为代表点，不是照片位置</span>
      </div>
      <template v-else>
        <div class="panel-hint">
          {{ cityCountryIso }} 城市候选代表点 · 非照片定位 · 共 {{ cityCount }} 个
        </div>
        <article v-for="city in cityScores" :key="city.city_id" class="candidate-card city-card">
          <div class="candidate-main">
            <span class="candidate-rank city">城</span>
            <div>
              <div class="candidate-name">{{ city.name }}</div>
              <div class="candidate-score">{{ city.region }} · {{ cityScoreLabel(city) }}</div>
              <div v-if="cityMatchedClueCount(city)" class="candidate-score city-clues">
                命中线索 {{ cityMatchedClueCount(city) }} 条（排序诊断，非概率）
              </div>
            </div>
          </div>
        </article>
      </template>
    </div>

    <div v-else class="candidate-body reference-tab">
      <ReferenceMatches :scores="scores" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { CountryScore } from '@/types'
import { normalizeCountryIso } from '@/utils/geo'
import {
  REFERENCE_MATCH_DISCLAIMER,
  REFERENCE_SIMILARITY_SEMANTICS,
  collectReferenceRegions,
  formatReferenceCosine,
  hasReferenceField,
  referenceCosineRows,
  topReferenceMatches,
} from '@/utils/reference'

const props = defineProps<{
  scores: CountryScore[]
}>()

const matches = computed(() => topReferenceMatches(props.scores))
const regions = computed(() => collectReferenceRegions(props.scores, 5))
const cosineRows = computed(() => referenceCosineRows(props.scores))
const hasReferenceData = computed(() => hasReferenceField(props.scores))
const hasAnyEvidence = computed(
  () => matches.value.length > 0 || regions.value.length > 0 || cosineRows.value.length > 0,
)
const isEmptyLibrary = computed(() => !hasReferenceData.value)
</script>

<template>
  <section class="reference-panel" aria-label="参考街景">
    <h3 class="reference-title">参考街景</h3>
    <p class="reference-disclaimer">
      {{ REFERENCE_MATCH_DISCLAIMER }}
    </p>

    <p v-if="!scores.length" class="reference-empty">
      上传照片后显示参考街景。
    </p>
    <p v-else-if="isEmptyLibrary" class="reference-empty">
      还没有可用的参考图。
    </p>
    <p v-else-if="!hasAnyEvidence" class="reference-empty">
      这次没有匹配到参考街景。
    </p>

    <template v-else>
      <div v-if="matches.length" class="reference-subtitle">参考图（前 {{ matches.length }} 项）</div>
      <ol v-if="matches.length" class="reference-list">
        <li v-for="match in matches" :key="match.reference_id" class="reference-item">
          <div class="reference-line">
            <span class="reference-place">{{ match.place || normalizeCountryIso(match.iso) }}</span>
            <span class="reference-cosine">{{ formatReferenceCosine(match.raw_cosine) }}</span>
          </div>
          <div class="reference-meta">
            {{ normalizeCountryIso(match.iso) }}
            <template v-if="match.city"> · {{ match.city }}</template>
            <template v-else-if="match.region"> · {{ match.region }}</template>
            <template v-if="match.viewpoint"> · {{ match.viewpoint }}</template>
          </div>
        </li>
      </ol>

      <div v-if="regions.length" class="reference-subtitle">按地区比较</div>
      <div v-if="regions.length" class="reference-regions">
        <div v-for="region in regions" :key="`${region.iso}-${region.region}`" class="reference-region">
          <span class="reference-place">{{ normalizeCountryIso(region.iso) }} · {{ region.region }}</span>
          <span class="reference-region-meta">
            {{ formatReferenceCosine(region.raw_cosine) }} · 参考图 {{ region.prototype_count }} 张
          </span>
        </div>
      </div>

      <div v-if="!regions.length && cosineRows.length" class="reference-regions">
        <div v-for="row in cosineRows" :key="row.iso" class="reference-region">
          <span class="reference-place">{{ row.name_zh }}</span>
          <span class="reference-region-meta">{{ formatReferenceCosine(row.cosine) }}</span>
        </div>
      </div>

      <p class="reference-semantics">{{ REFERENCE_SIMILARITY_SEMANTICS }}</p>
    </template>
  </section>
</template>

<style scoped>
.reference-panel {
  flex-shrink: 0;
  margin-top: 0;
  padding-top: 0;
  border-top: 0;
  color: var(--ink-soft);
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
}
.reference-title { margin-bottom: 3px; font-size: 15px;
  line-height: 22px; font-weight: 600; color: var(--ink); }
.reference-disclaimer { margin-bottom: 5px; font-size: 13px;
  font-weight: 400; line-height: 20px; color: var(--faint); }
.reference-empty { font-size: 13px;
  font-weight: 400; line-height: 20px; color: var(--muted); }
.reference-subtitle {
  margin: 8px 0 4px;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.reference-list { display: flex; flex-direction: column; gap: 8px; list-style: none; }
.reference-item {
  padding: 16px;
  border-left: 1px solid var(--blue-line);
  border-radius: 0;
  background: rgba(255, 255, 255, 0.018);
}
.reference-line { display: flex; justify-content: space-between; gap: 8px; }
.reference-place { font-size: 15px;
  line-height: 22px; font-weight: 600; color: var(--ink-soft); }
.reference-cosine { color: var(--faint); font-family: var(--font-mono); font-size: 13px;
  font-weight: 400;
  line-height: 20px; white-space: nowrap; }
.reference-meta { display: flex; flex-wrap: wrap; gap: 3px; color: var(--faint); font-family: var(--font-mono); font-size: 13px;
  font-weight: 400;
  line-height: 20px; }
.reference-id { opacity: 0.7; }
.reference-regions { display: flex; flex-direction: column; gap: 3px; }
.reference-region { display: flex; justify-content: space-between; gap: 8px; font-size: 13px;
  font-weight: 400;
  line-height: 20px; }
.reference-region-meta { color: var(--faint); font-family: var(--font-mono); font-size: 13px;
  font-weight: 400;
  line-height: 20px; text-align: right; }
.reference-semantics { margin-top: 7px; font-size: 13px;
  font-weight: 400; line-height: 20px; color: var(--faint); }
</style>

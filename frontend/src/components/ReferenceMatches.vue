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

/** Raw support/prototype coverage metric; never rendered as a probability. */
function formatSupport(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return Number.isInteger(value) ? String(value) : value.toFixed(3)
}
</script>

<template>
  <section class="reference-panel" aria-label="参考街景相似结果">
    <h3 class="reference-title">参考街景相似结果</h3>
    <p class="reference-disclaimer">
      {{ REFERENCE_MATCH_DISCLAIMER }}
    </p>

    <p v-if="!scores.length" class="reference-empty">
      上传图片后，如有参考街景诊断会显示在这里。
    </p>
    <p v-else-if="isEmptyLibrary" class="reference-empty">
      未配置带标签的参考图库，暂无参考街景诊断。
    </p>
    <p v-else-if="!hasAnyEvidence" class="reference-empty">
      已配置参考图库，本次未返回可用的参考街景匹配。
    </p>

    <template v-else>
      <div v-if="matches.length" class="reference-subtitle">Top {{ matches.length }} 参考图匹配</div>
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
            <span class="reference-id">#{{ match.reference_id }}</span>
          </div>
        </li>
      </ol>

      <div v-if="regions.length" class="reference-subtitle">国家/地区原型分数</div>
      <div v-if="regions.length" class="reference-regions">
        <div v-for="region in regions" :key="`${region.iso}-${region.region}`" class="reference-region">
          <span class="reference-place">{{ normalizeCountryIso(region.iso) }} · {{ region.region }}</span>
          <span class="reference-region-meta">
            {{ formatReferenceCosine(region.raw_cosine) }} · 原型 {{ region.prototype_count }} ·
            支持 {{ formatSupport(region.support) }}
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
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
  font-size: 12px;
  color: var(--text);
}
.reference-title { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
.reference-disclaimer { font-size: 11px; color: var(--warn); margin-bottom: 4px; }
.reference-empty { font-size: 11px; color: var(--muted); }
.reference-subtitle { font-size: 11px; font-weight: 700; color: var(--primary); margin: 6px 0 4px; }
.reference-list { list-style: none; display: flex; flex-direction: column; gap: 4px; }
.reference-item {
  background: #f8fafc;
  border-left: 3px solid var(--primary);
  border-radius: 4px;
  padding: 5px 8px;
}
.reference-line { display: flex; justify-content: space-between; gap: 8px; }
.reference-place { font-weight: 600; }
.reference-cosine { color: var(--muted); font-size: 11px; white-space: nowrap; }
.reference-meta { color: var(--muted); font-size: 11px; display: flex; flex-wrap: wrap; gap: 4px; }
.reference-id { opacity: 0.7; }
.reference-regions { display: flex; flex-direction: column; gap: 3px; }
.reference-region { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; }
.reference-region-meta { color: var(--muted); text-align: right; }
.reference-semantics { margin-top: 6px; font-size: 11px; color: var(--muted); }
</style>

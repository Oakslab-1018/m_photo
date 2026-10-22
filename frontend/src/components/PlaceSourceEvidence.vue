<script setup lang="ts">
import { computed } from 'vue'
import type { PlaceSource } from '@/types'
import {
  normalizePlaceSources,
  observationSourceLabel,
  placeSourceDisplay,
  placeSourceTypeLabel,
  type PlaceSourceDisplay,
} from '@/utils/placeSources'

const props = withDefaults(
  defineProps<{
    sources?: readonly PlaceSource[] | null
    compact?: boolean
  }>(),
  {
    sources: () => [],
    compact: false,
  },
)

const normalizedSources = computed(() => normalizePlaceSources(props.sources))

function uniqueLabels(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))]
}

const retrievalLabels = computed(() =>
  uniqueLabels(normalizedSources.value.map(placeSourceTypeLabel)),
)
const observationLabels = computed(() =>
  uniqueLabels(normalizedSources.value.map(observationSourceLabel)),
)
const rows = computed(() =>
  normalizedSources.value
    .map(placeSourceDisplay)
    .filter(
      (row): row is PlaceSourceDisplay =>
        row !== null &&
        Boolean(row.title || row.label || row.excerpt || row.fetchedAt || row.url),
    ),
)
</script>

<template>
  <div
    v-if="normalizedSources.length"
    class="place-source-evidence"
    :class="{ 'place-source-evidence--compact': compact }"
  >
    <div v-if="retrievalLabels.length" class="place-source-provenance">
      <span class="place-source-provenance-label">检索来源：</span>
      <span
        v-for="label in retrievalLabels"
        :key="label"
        class="place-source-badge"
      >{{ label }}</span>
    </div>
    <div v-if="observationLabels.length" class="place-source-provenance">
      <span class="place-source-provenance-label">文字来源：</span>
      <span
        v-for="label in observationLabels"
        :key="label"
        class="place-source-badge place-source-badge--text"
      >{{ label }}</span>
    </div>

    <ul v-if="rows.length" class="place-source-list">
      <li v-for="row in rows" :key="row.key" class="place-source-row">
        <div class="place-source-row-line">
          <span v-if="row.title" class="place-source-row-title">{{ row.title }}</span>
          <span v-else-if="row.label" class="place-source-row-title">{{ row.label }}</span>
          <a
            v-if="row.url"
            :href="row.url"
            class="place-source-link"
            target="_blank"
            rel="noopener noreferrer"
          >查看来源</a>
        </div>
        <p v-if="row.excerpt" class="place-source-excerpt">{{ row.excerpt }}</p>
        <small v-if="row.fetchedAt">获取时间：{{ row.fetchedAt }}</small>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.place-source-evidence {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.place-source-provenance {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 3px 5px;
}

.place-source-provenance-label {
  color: var(--faint);
  font-size: 12px;
  line-height: 18px;
}

.place-source-badge {
  padding: 1px 6px;
  border: 1px solid var(--guided-line, var(--line));
  border-radius: 999px;
  background: var(--guided-panel, rgba(255, 255, 255, 0.025));
  color: var(--guided-muted, var(--muted));
  font-size: 12px;
  line-height: 16px;
  overflow-wrap: anywhere;
}

.place-source-badge--text {
  border-style: dashed;
}

.place-source-list {
  display: grid;
  gap: 5px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.place-source-row {
  display: grid;
  gap: 2px;
  min-width: 0;
  padding: 5px 7px;
  border-left: 2px solid var(--guided-line, var(--line));
}

.place-source-row-line {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.place-source-row-title,
.place-source-excerpt,
.place-source-row small {
  min-width: 0;
  color: var(--guided-muted, var(--muted));
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
  overflow-wrap: anywhere;
}

.place-source-row-title {
  color: var(--guided-ink, var(--ink));
}

.place-source-link {
  flex: 0 0 auto;
  color: var(--guided-teal, var(--amber));
  font-size: 12px;
  line-height: 18px;
  text-underline-offset: 2px;
}

.place-source-link:hover {
  color: var(--guided-teal-dark, var(--amber-bright));
}

.place-source-row small {
  color: var(--faint);
  font-family: var(--font-mono);
}

.place-source-evidence--compact .place-source-row {
  padding: 3px 6px;
}

.place-source-evidence--compact .place-source-excerpt {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  overflow: hidden;
}
</style>

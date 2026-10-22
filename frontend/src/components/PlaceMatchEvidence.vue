<script setup lang="ts">
import { computed } from 'vue'
import PlaceSourceEvidence from '@/components/PlaceSourceEvidence.vue'
import type { PlaceMatch } from '@/types'
import { normalizePlaceMatch, placeMatchIsCandidateOnly, placeMatchVerificationLabel } from '@/utils/placeSources'

const props = defineProps<{ match: PlaceMatch }>()

const match = computed(() => normalizePlaceMatch(props.match))
const candidateOnly = computed(() => placeMatchIsCandidateOnly(match.value))
const queryTerms = computed(() => match.value?.query_terms ?? [])
const attribution = computed(() =>
  [
    ...new Set(
      (match.value?.sources ?? [])
        .map((source) => source.attribution?.trim())
        .filter((value): value is string => Boolean(value)),
    ),
  ].join(' · '),
)
</script>

<template>
  <div v-if="match" class="place-evidence" :class="{ 'place-evidence--candidate': candidateOnly }">
    <span v-if="candidateOnly" class="place-evidence-status">{{ placeMatchVerificationLabel(match) }}（不作为地名核验）</span>
    <span v-if="match.ambiguous">同名地点，需补充城市或区名后确认</span>
    <span v-if="queryTerms.length">文字线索：{{ queryTerms.join('、') }}</span>
    <PlaceSourceEvidence :sources="match.sources" />
    <small v-if="attribution">{{ attribution }}</small>
  </div>
</template>

<style scoped>
.place-evidence {
  display: grid;
  gap: 3px;
  margin-top: 5px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.place-evidence-status {
  color: var(--guided-candidate, var(--amber));
  font-size: 12px;
  line-height: 18px;
}

.place-evidence small {
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  overflow-wrap: anywhere;
}
</style>

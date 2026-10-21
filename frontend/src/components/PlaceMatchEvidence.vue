<script setup lang="ts">
import { computed } from 'vue'
import type { PlaceMatch } from '@/types'
const props = defineProps<{ match: PlaceMatch }>()
const attribution = computed(() => [...new Set(props.match.sources.map(source => source.attribution).filter(Boolean))].join(' · '))
</script>

<template>
  <div class="place-evidence">
    <span v-if="match.ambiguous">同名地点，需补充城市或区名后确认</span>
    <span v-else>文字线索：{{ match.query_terms.join('、') }}</span>
    <span v-for="source in match.sources" :key="`${source.provider}:${source.provider_place_id ?? source.label}`">{{ source.label }}</span>
    <small v-if="attribution">{{ attribution }}</small>
  </div>
</template>

<style scoped>
.place-evidence { display: grid; gap: 3px; margin-top: 6px; color: var(--text-secondary, #657080); font-size: 12px; line-height: 1.5; }
</style>

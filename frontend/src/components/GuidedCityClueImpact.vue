<script setup lang="ts">
import { computed } from 'vue'
import PlaceMatchEvidence from '@/components/PlaceMatchEvidence.vue'
import PlaceSourceEvidence from '@/components/PlaceSourceEvidence.vue'
import type { LocationCandidate, PlaceSource } from '@/types'
import { clueValueLabel, dimensionLabel } from '@/utils/clues'
import {
  guidedCityClueImpact,
  guidedSignedValue,
  safeGuidedText,
  type GuidedMatchedClueImpact,
} from '@/utils/guided'
import {
  legacyPlaceSourceLabel,
  placeMatchIsCandidateOnly,
  placeMatchVerificationLabel,
} from '@/utils/placeSources'

const props = defineProps<{ candidate: LocationCandidate }>()

const impact = computed(() => guidedCityClueImpact(props.candidate))
const hasFusionBreakdown = computed(() =>
  Boolean(
    impact.value &&
      [impact.value.placeBoost, impact.value.sceneDelta].some((value) => value !== null),
  ),
)
const candidateIsPlaceCandidate = computed(() =>
  placeMatchIsCandidateOnly(props.candidate.place_match),
)
const hasSceneModelSignal = computed(() => {
  const current = impact.value
  if (!current) return false
  const signals = [current.signalKind, ...current.matchedClues.map((clue) => clue.signalKind)]
  return signals.some((signal) => evidenceCategory(signal) === 'scene')
})

function evidenceCategory(signal: string, source = ''): 'scene' | 'place' | '' {
  const normalizedSignal = signal.trim().toLowerCase()
  if (normalizedSignal === 'model_similarity' || normalizedSignal.includes('scene')) {
    return 'scene'
  }
  const evidenceKey = `${normalizedSignal} ${source.trim().toLowerCase()}`
  if (/(place|geocod|catalog|address)/.test(evidenceKey)) return 'place'
  return ''
}

function sourcesAreCandidateOnly(sources: readonly PlaceSource[]): boolean {
  if (!sources.length) return false
  return placeMatchIsCandidateOnly({
    query_terms: [],
    clue_ids: [],
    match_kind: 'catalog_exact',
    ambiguous: false,
    sources: [...sources],
  })
}

function clueEvidenceLabel(clue: GuidedMatchedClueImpact): string {
  const sourceKey = [
    clue.source,
    ...clue.sourceRefs.flatMap((source) => [
      source.source_type ?? '',
      source.provider ?? '',
      source.url ?? '',
    ]),
  ].join(' ')
  const category = evidenceCategory(clue.signalKind, sourceKey)
  if (category === 'scene') return '模型观察（仅参与排序）'
  if (category === 'place') {
    return candidateIsPlaceCandidate.value || sourcesAreCandidateOnly(clue.sourceRefs)
      ? '地点候选 · 待验证'
      : '地名证据匹配'
  }
  return ''
}

function placeBoostLabel(): string {
  if (!impact.value || impact.value.placeBoostIsLegacy) return '地名线索调整'
  return candidateIsPlaceCandidate.value ? '候选地名调整' : '地名证据调整'
}

function clueLabel(clue: GuidedMatchedClueImpact): string {
  const dimension = clue.dimension ? dimensionLabel(clue.dimension) : '其他线索'
  const value = clue.valueLabel?.trim() || (clue.value ? clueValueLabel(clue.value) : '')
  return value ? `${dimension} · ${value}` : dimension
}

function clueKey(clue: GuidedMatchedClueImpact, index: number): string {
  return [
    index,
    clue.dimension,
    clue.value,
    clue.valueLabel,
    clue.gain,
    clue.evidence,
    clue.source,
    clue.sourceRefs
      .map((source) =>
        [
          source.source_type,
          source.url,
          source.title,
          source.excerpt,
          source.fetched_at,
          source.observation_source,
        ].join('|'),
      )
      .join(';'),
    clue.signalKind,
  ].join(':')
}

function signedClass(value: number | null): string {
  if (value === null || value === 0) return 'neutral'
  return value > 0 ? 'positive' : 'negative'
}
</script>

<template>
  <details v-if="impact?.hasDetail" class="guided-city-impact">
    <summary>查看线索如何影响排序</summary>
    <div class="guided-city-impact-body">
      <p v-if="hasSceneModelSignal" class="guided-city-impact-note">
        场景条件仅表示图像与固定场景描述相似，不是城市属性核验。
      </p>

      <section v-if="impact.matchedClues.length" class="guided-city-impact-section">
        <h4>逐维线索贡献</h4>
        <ul class="guided-city-impact-list">
          <li
            v-for="(clue, index) in impact.matchedClues"
            :key="clueKey(clue, index)"
            class="guided-city-impact-row"
          >
            <div class="guided-city-impact-line">
              <span class="guided-city-impact-label">{{ clueLabel(clue) }}</span>
              <span
                v-if="clue.gain !== null"
                class="guided-city-impact-gain"
                :class="signedClass(clue.gain)"
              >{{ guidedSignedValue(clue.gain) }}</span>
              <span v-else class="guided-city-impact-missing">未提供贡献值</span>
            </div>
            <p v-if="clue.evidence" class="guided-city-impact-evidence">
              {{ safeGuidedText(clue.evidence) }}
            </p>
            <small v-if="clueEvidenceLabel(clue)">
              证据类别：{{ clueEvidenceLabel(clue) }}
            </small>
            <PlaceSourceEvidence
              v-if="clue.sourceRefs.length"
              :sources="clue.sourceRefs"
              compact
            />
            <small v-else-if="legacyPlaceSourceLabel(clue.source)">
              {{ legacyPlaceSourceLabel(clue.source) }}
            </small>
          </li>
        </ul>
      </section>

      <dl v-if="hasFusionBreakdown" class="guided-city-impact-breakdown">
        <template v-if="impact.placeBoost !== null">
          <dt>{{ placeBoostLabel() }}</dt>
          <dd :class="signedClass(impact.placeBoost)">{{ guidedSignedValue(impact.placeBoost) }}</dd>
        </template>
        <template v-if="impact.sceneDelta !== null">
          <dt>场景条件模型匹配 Δ</dt>
          <dd :class="signedClass(impact.sceneDelta)">{{ guidedSignedValue(impact.sceneDelta) }}</dd>
        </template>
      </dl>

      <section v-if="candidate.place_match" class="guided-city-impact-section">
        <h4>{{ placeMatchVerificationLabel(candidate.place_match) }}</h4>
        <PlaceMatchEvidence :match="candidate.place_match" />
      </section>

      <section
        v-if="hasSceneModelSignal || impact.sceneVersion"
        class="guided-city-impact-section"
      >
        <h4>模型观察（仅参与排序）</h4>
        <p v-if="impact.sceneVersion" class="guided-city-impact-evidence">
          场景模板版本：{{ safeGuidedText(impact.sceneVersion) }}
        </p>
        <small>模型相似度只参与排序，不证实城市具有山地、河流或建筑等属性。</small>
      </section>

      <section v-if="impact.clueNotes.length" class="guided-city-impact-section">
        <h4>未计分原因</h4>
        <p
          v-for="note in impact.clueNotes"
          :key="note"
          class="guided-city-impact-evidence"
        >{{ safeGuidedText(note) }}</p>
      </section>
    </div>
  </details>
</template>

<style scoped>
.guided-city-impact {
  margin-left: 36px;
  border-top: 1px dashed var(--guided-line, var(--line));
}

.guided-city-impact summary {
  min-height: 32px;
  padding: 6px 0 2px;
  color: var(--guided-teal, var(--amber));
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  cursor: pointer;
  list-style: none;
}

.guided-city-impact summary::-webkit-details-marker {
  display: none;
}

.guided-city-impact summary:hover {
  color: var(--guided-teal-dark, var(--amber-bright));
  text-decoration: underline;
  text-underline-offset: 3px;
}

.guided-city-impact-body {
  display: grid;
  gap: 10px;
  padding: 3px 0 6px;
}

.guided-city-impact-note,
.guided-city-impact-evidence {
  color: var(--guided-muted, var(--muted));
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  overflow-wrap: anywhere;
}

.guided-city-impact-note {
  padding: 6px 8px;
  border-left: 2px solid var(--amber-line, var(--line-strong));
  background: var(--amber-soft, rgba(255, 255, 255, 0.03));
}

.guided-city-impact-section {
  display: grid;
  gap: 4px;
}

.guided-city-impact-section h4 {
  color: var(--guided-ink, var(--ink));
  font-size: 13px;
  font-weight: 600;
  line-height: 20px;
}

.guided-city-impact-list {
  display: grid;
  gap: 5px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.guided-city-impact-row {
  display: grid;
  gap: 1px;
  padding: 6px 8px;
  border: 1px solid var(--guided-line, var(--line));
  border-radius: 4px;
}

.guided-city-impact-line {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}

.guided-city-impact-label {
  min-width: 0;
  color: var(--guided-ink, var(--ink));
  font-size: 13px;
  line-height: 20px;
  overflow-wrap: anywhere;
}

.guided-city-impact-gain,
.guided-city-impact-breakdown dd,
.guided-city-impact-missing {
  flex: 0 0 auto;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}

.guided-city-impact-gain.positive,
.guided-city-impact-breakdown dd.positive {
  color: var(--sage, #83b98d);
}

.guided-city-impact-gain.negative,
.guided-city-impact-breakdown dd.negative {
  color: var(--guided-danger, var(--coral));
}

.guided-city-impact-gain.neutral,
.guided-city-impact-breakdown dd.neutral,
.guided-city-impact-missing {
  color: var(--faint);
}


.guided-city-impact-row small,
.guided-city-impact-section > small {
  color: var(--faint);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
  overflow-wrap: anywhere;
}

.guided-city-impact-breakdown {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 3px 8px;
  margin: 0;
  padding: 6px 8px;
  border: 1px solid var(--guided-line, var(--line));
  border-radius: 4px;
}

.guided-city-impact-breakdown dt {
  color: var(--guided-muted, var(--muted));
  font-size: 13px;
  line-height: 20px;
}

.guided-city-impact-breakdown dd {
  margin: 0;
  text-align: right;
}

@media (max-width: 480px) {
  .guided-city-impact {
    margin-left: 0;
  }
}
</style>
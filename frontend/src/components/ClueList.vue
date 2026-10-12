<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { Clue } from '@/types'
import {
  clueIdentity,
  cluePreview,
  clueProvenance,
  clueSourceLabel,
  clueValueLabel,
  confidencePercent,
  defaultOpenDimensions,
  groupClues,
  isCatalogValidated,
  isLowConfidence,
  isReferenceOnly,
  type ClueGroup,
} from '@/utils/clues'

const props = defineProps<{
  clues: Clue[]
  loading?: boolean
  /** Newly appended clues that should be visible even when a group is capped. */
  recentClueKeys?: string[]
}>()

/** OCR can emit one clue per line; long groups stay capped behind an expand button. */
const ITEM_LIMIT = 4

const groups = computed(() => groupClues(props.clues))
const totalCount = computed(() => groups.value.reduce((total, group) => total + group.items.length, 0))
const listRef = ref<HTMLElement | null>(null)
const recentKeySet = computed(() => new Set(props.recentClueKeys ?? []))
const openState = ref<Record<string, boolean>>({})
const expanded = ref<Record<string, boolean>>({})

function recentDimensions(
  current: readonly ClueGroup[],
  keys: readonly string[] = props.recentClueKeys ?? [],
): Set<string> {
  const recent = new Set(keys)
  const dimensions = new Set<string>()
  for (const group of current) {
    if (group.items.some((clue) => recent.has(clueIdentity(clue.dimension, clue.value)))) {
      dimensions.add(group.dimension)
    }
  }
  return dimensions
}

function isRecentClue(clue: Clue): boolean {
  return recentKeySet.value.has(clueIdentity(clue.dimension, clue.value))
}

function scrollToLatestRecent(keys: readonly string[]): void {
  const container = listRef.value
  if (!container) return
  const cards = Array.from(container.querySelectorAll<HTMLElement>('[data-clue-key]'))
  for (const key of [...keys].reverse()) {
    const target = cards.find((card) => card.dataset.clueKey === key)
    if (!target) continue
    const containerRect = container.getBoundingClientRect()
    const targetRect = target.getBoundingClientRect()
    if (targetRect.top < containerRect.top || targetRect.bottom > containerRect.bottom) {
      container.scrollTop +=
        targetRect.top - containerRect.top - Math.max(0, (containerRect.height - targetRect.height) / 2)
    }
    break
  }
}

// Open the most actionable group (user evidence first, then strongest clue) while
// preserving any manual toggle the user already made for an existing dimension.
watch(
  groups,
  (current) => {
    const defaults = new Set(defaultOpenDimensions(current))
    const recent = recentDimensions(current)
    const next: Record<string, boolean> = {}
    for (const group of current) {
      next[group.dimension] =
        openState.value[group.dimension] ?? (defaults.has(group.dimension) || recent.has(group.dimension))
    }
    openState.value = next
  },
  { immediate: true },
)

watch(
  () => [...(props.recentClueKeys ?? [])],
  async (current, previous) => {
    const previousKeys = new Set(previous ?? [])
    const added = current.filter((key) => !previousKeys.has(key))
    if (!added.length) return
    const dimensions = recentDimensions(groups.value, added)
    if (!dimensions.size) return
    const nextOpen = { ...openState.value }
    const nextExpanded = { ...expanded.value }
    for (const dimension of dimensions) {
      nextOpen[dimension] = true
      nextExpanded[dimension] = true
    }
    openState.value = nextOpen
    expanded.value = nextExpanded
    await nextTick()
    scrollToLatestRecent(added)
  },
  { immediate: true, flush: 'post' },
)

function onToggle(dimension: string, event: Event): void {
  const target = event.target as HTMLDetailsElement | null
  if (!target || typeof target.open !== 'boolean') return
  openState.value = { ...openState.value, [dimension]: target.open }
}

function previewFor(group: ClueGroup): { visible: Clue[]; hiddenCount: number } {
  const limit = expanded.value[group.dimension] ? group.items.length : ITEM_LIMIT
  return cluePreview(group.items, limit)
}

function toggleExpanded(dimension: string): void {
  expanded.value = { ...expanded.value, [dimension]: !expanded.value[dimension] }
}
</script>

<template>
  <div ref="listRef" class="clue-list" aria-live="polite">
    <div v-if="loading" class="empty-hint">正在提取多维度线索…</div>
    <div v-else-if="!groups.length" class="empty-hint">
      上传图片后，AI 将自动提取多维度线索，也可在对话中手动补充。
    </div>
    <template v-else>
      <div class="clue-overview">
        <span><strong>{{ groups.length }}</strong> 个维度 · <strong>{{ totalCount }}</strong> 条线索</span>
        <span class="clue-overview-hint">按证据强度排序</span>
      </div>
      <details
        v-for="group in groups"
        :key="group.dimension"
        class="clue-group"
        :open="!!openState[group.dimension]"
        @toggle="onToggle(group.dimension, $event)"
      >
        <summary>
          <span class="clue-group-name">{{ group.label }}</span>
          <span class="badge">{{ group.items.length }}</span>
        </summary>
        <div class="clue-group-items">
          <article
            v-for="clue in previewFor(group).visible"
            :key="`${clue.dimension}-${clue.value}`"
            class="clue-card"
            :class="{ low: isLowConfidence(clue.confidence), 'is-new': isRecentClue(clue) }"
            :data-clue-key="clueIdentity(clue.dimension, clue.value)"
          >
            <div class="val">
              {{ clueValueLabel(clue.value) }}
              <span v-if="isRecentClue(clue)" class="tag new">新线索</span>
            </div>
            <div class="conf">
              <span class="conf-bar" aria-hidden="true">
                <span class="conf-fill" :style="{ width: confidencePercent(clue.confidence) }" />
              </span>
              <span>置信度 {{ confidencePercent(clue.confidence) }}</span>
              <span
                v-if="clueSourceLabel(clue)"
                class="tag source"
                :data-provenance="clueProvenance(clue)"
                :title="clue.source || undefined"
              >{{ clueSourceLabel(clue) }}</span>
              <span v-if="isReferenceOnly(clue)" class="tag reference-only" title="仅作观察展示，不参与排序">仅供参考</span>
              <span v-else-if="isCatalogValidated(clue)" class="tag validated" title="已按国家目录核对">目录已核对</span>
            </div>
            <div v-if="clue.evidence" class="ev">{{ clue.evidence }}</div>
          </article>
          <button
            v-if="previewFor(group).hiddenCount"
            type="button"
            class="clue-more"
            @click="toggleExpanded(group.dimension)"
          >
            展开其余 {{ previewFor(group).hiddenCount }} 条
          </button>
          <button
            v-else-if="expanded[group.dimension] && group.items.length > ITEM_LIMIT"
            type="button"
            class="clue-more"
            @click="toggleExpanded(group.dimension)"
          >
            收起
          </button>
        </div>
      </details>
    </template>
  </div>
</template>
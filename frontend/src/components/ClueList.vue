<script setup lang="ts">
import { computed } from 'vue'
import type { Clue } from '@/types'
import { clueValueLabel, confidencePercent, groupClues } from '@/utils/clues'

const props = defineProps<{
  clues: Clue[]
  loading?: boolean
}>()

const groups = computed(() => groupClues(props.clues))
</script>

<template>
  <div class="clue-list" aria-live="polite">
    <div v-if="loading" class="empty-hint">正在提取多维度线索…</div>
    <div v-else-if="!groups.length" class="empty-hint">
      上传图片后，AI 将自动提取多维度线索，也可在对话中手动补充。
    </div>
    <template v-else>
      <section v-for="group in groups" :key="group.dimension" class="clue-group">
        <h3 class="clue-group-title">{{ group.label }}</h3>
        <div class="clue-group-items">
          <article v-for="clue in group.items" :key="`${clue.dimension}-${clue.value}`" class="clue-card">
            <div class="val">{{ clueValueLabel(clue.value) }}</div>
            <div class="conf">
              置信度 {{ confidencePercent(clue.confidence) }}
              <span v-if="clue.source === 'user'" class="source-tag">用户补充</span>
              <span v-else-if="clue.source === 'point_click'" class="source-tag">局部分析</span>
            </div>
            <div v-if="clue.evidence" class="ev">{{ clue.evidence }}</div>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>

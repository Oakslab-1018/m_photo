<script setup lang="ts">
import type { Ref } from 'vue'
import type { FocusMode } from '@/components/uiTypes'

const props = defineProps<{
  backgroundInert: boolean
  hasSession: boolean
  busy: boolean
  sessionStatus: string
  liveTone: 'ok' | 'muted' | 'warn' | 'info'
  countryBreadcrumb: string
  cityBreadcrumb: string
  focusMode: FocusMode
  historyOpen: boolean
  historyAllowed: boolean
  rootRef: Ref<HTMLElement | null>
}>()

const emit = defineEmits<{
  'focus-mode': [mode: FocusMode]
  'open-history': []
  'new-analysis': []
}>()

const FOCUS_MODES: Array<{ key: FocusMode; label: string; hint: string }> = [
  { key: 'collab', label: '全部', hint: '照片、地图和助手同时显示' },
  { key: 'map', label: '地图', hint: '只看地图' },
  { key: 'photo', label: '照片', hint: '只看照片和线索' },
]
</script>

<template>
  <header :ref="props.rootRef" class="command-bar" aria-label="地理定位命令栏" :inert="backgroundInert">
    <div class="command-brand">
      <span class="command-brand__mark" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">
          <circle cx="12" cy="12" r="8.4" />
          <path d="M3.8 12h16.4M12 3.6c2.4 2.5 3.6 5.3 3.6 8.4S14.4 17.9 12 20.4c-2.4-2.5-3.6-5.3-3.6-8.4S9.6 6.1 12 3.6z" />
        </svg>
      </span>
      <span class="command-brand__copy">
        <h1 class="command-brand__title">M-PHOTO</h1>
        <span class="command-brand__sub">照片定位助手</span>
      </span>
    </div>

    <nav class="session-breadcrumb" aria-label="当前定位进度">
      <span class="breadcrumb-step">
        <span class="breadcrumb-index">01</span>
        <span class="sr-only">国家</span>
        <strong :title="countryBreadcrumb">{{ countryBreadcrumb }}</strong>
      </span>
      <span class="breadcrumb-arrow" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M5 12h14M14 7l5 5-5 5" />
        </svg>
      </span>
      <span class="breadcrumb-step breadcrumb-step--city">
        <span class="breadcrumb-index">02</span>
        <span class="sr-only">城市</span>
        <strong :title="cityBreadcrumb">{{ cityBreadcrumb }}</strong>
      </span>
    </nav>

    <div class="command-session" :class="liveTone" :title="sessionStatus">
      <span class="session-dot" :class="{ active: hasSession, busy }" />
      <span class="command-session__text">{{ sessionStatus }}</span>
    </div>

    <div class="focus-switch" role="group" aria-label="界面布局">
      <button
        v-for="mode in FOCUS_MODES"
        :key="mode.key"
        type="button"
        class="focus-mode-btn"
        :class="{ active: focusMode === mode.key }"
        :aria-pressed="focusMode === mode.key"
        :aria-label="mode.label"
        :title="mode.hint"
        @click="emit('focus-mode', mode.key)"
      >
        <svg v-if="mode.key === 'collab'" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <path d="M8 9h6M8 13h4M6 18l-3 2 1-4a7 7 0 1 1 12 0" />
          <path d="M13 10h5M13 14h3M18 18l3 2-1-4a6 6 0 0 0-4-5" />
        </svg>
        <svg v-else-if="mode.key === 'map'" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <path d="m3 6 5-2 8 3 5-2v14l-5 2-8-3-5 2zM8 4v14M16 7v14" />
        </svg>
        <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <rect x="3" y="4" width="18" height="16" rx="2.5" />
          <circle cx="9" cy="9" r="1.7" />
          <path d="m21 15-5-5L6 20" />
        </svg>
        <span class="focus-label">{{ mode.label }}</span>
      </button>
    </div>

    <button
      type="button"
      class="command-action"
      :class="{ active: historyOpen }"
      :aria-expanded="historyOpen"
      aria-controls="reasoning-history-drawer"
      :disabled="!historyAllowed"
      :title="historyAllowed ? '打开对话记录' : '核对完成并查看候选后可打开对话记录'"
      @click="emit('open-history')"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
        <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
        <path d="M3 3v5h5M12 7v5l3 2" />
      </svg>
      <span class="command-action__label">对话</span>
    </button>

    <button type="button" class="command-action command-action--strong" @click="emit('new-analysis')">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
        <path d="M12 5v14M5 12h14" />
      </svg>
      <span class="command-action__label">新建</span>
    </button>
  </header>
</template>

<script setup lang="ts">
import { computed, type Ref } from 'vue'
import { AnimatePresence, motion, useReducedMotion } from 'motion-v'
import CandidatePanel from '@/components/CandidatePanel.vue'
import ClueList from '@/components/ClueList.vue'
import UploadPanel from '@/components/UploadPanel.vue'
import type { Clue, Point } from '@/types'
import type { CandidatePanelBindings, EvidenceTab } from '@/components/uiTypes'

const props = defineProps<{
  visible: boolean
  collapsed: boolean
  isMobile: boolean
  mobileOpen: boolean
  inert: boolean
  imagePresent: boolean
  previewUrl: string | null
  clueCount: number
  regionReportCount: number
  clickPoints: Point[]
  clues: Clue[]
  recentClueKeys: string[]
  uploading: boolean
  uploadNotice: string | null
  selectedFileName: string | null
  hasSession: boolean
  hasLocalPreview: boolean
  candidates: CandidatePanelBindings
  guidedActive: boolean
  guidedCountryAllowed: boolean
  mobileTab: EvidenceTab
  rootRef: Ref<HTMLElement | null>
  dialogRef: Ref<HTMLElement | null>
}>()

const emit = defineEmits<{
  expand: []
  collapse: []
  close: []
  'open-tab': [tab: EvidenceTab]
  file: [file: File]
  upload: []
  'open-viewer': []
  'locate-clue': [clue: Clue]
}>()

const reduceMotion = useReducedMotion()
const enterFrom = computed(() => (reduceMotion.value ? false : { opacity: 0, y: 8 }))
const enterSettle = { opacity: 1, y: 0 }
const enterEase = { duration: 0.24, ease: [0.2, 0.7, 0.2, 1] as const }

</script>

<template>
  <AnimatePresence>
    <motion.div
      v-if="isMobile && mobileOpen"
      key="mobile-evidence-backdrop"
      class="mobile-evidence-backdrop"
      aria-hidden="true"
      :initial="reduceMotion ? false : { opacity: 0 }"
      :animate="{ opacity: 1 }"
      :exit="{ opacity: 0 }"
      :transition="{ duration: reduceMotion ? 0 : 0.18 }"
      @click="emit('close')"
    />
  </AnimatePresence>

  <aside
    v-show="visible"
    :ref="props.rootRef"
    class="dock-anchor evidence-dock-anchor"
    :class="{ 'is-mobile-open': isMobile && mobileOpen, 'is-collapsed': collapsed }"
    :inert="inert"
    aria-label="照片与线索"
  >
    <motion.button
      v-if="collapsed"
      type="button"
      class="dock-thumbnail evidence-thumbnail"
      aria-label="展开照片与线索"
      :initial="enterFrom"
      :animate="enterSettle"
      :transition="enterEase"
      @click="emit('expand')"
    >
      <span class="dock-thumbnail__media">
        <img v-if="previewUrl" :src="previewUrl" alt="" />
        <span v-else class="dock-thumbnail__empty" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <rect x="3" y="4" width="18" height="16" rx="2.5" />
            <path d="m21 15-5-5L6 20" />
          </svg>
        </span>
        <span class="dock-thumbnail__badge">{{ imagePresent ? 'PHOTO' : 'UPLOAD' }}</span>
      </span>
      <span class="dock-thumbnail__copy">
        <strong>照片与线索</strong>
        <small>{{ clueCount }} 条线索</small>
      </span>
    </motion.button>

    <section
      v-else
      :ref="props.dialogRef"
      class="floating-dock evidence-dock"
      :role="isMobile ? 'dialog' : undefined"
      :aria-modal="isMobile ? 'true' : undefined"
      :tabindex="isMobile ? -1 : undefined"
      aria-labelledby="evidence-dock-title"
    >
      <header class="dock-head">
        <div class="dock-heading">
          <span class="section-kicker">EVIDENCE</span>
          <h2 id="evidence-dock-title">照片与线索</h2>
        </div>
        <div class="dock-head__actions">
          <span v-if="imagePresent" class="dock-state-chip">{{ regionReportCount }} 次分析</span>
          <button
            v-if="!isMobile"
            type="button"
            class="icon-btn"
            aria-label="收起照片与线索"
            title="收起为缩略图"
            @click="emit('collapse')"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
              <path d="m15 18-6-6 6-6" />
            </svg>
          </button>
          <button
            v-else
            type="button"
            class="icon-btn"
            data-modal-autofocus
            aria-label="关闭照片与线索浮层"
            @click="emit('close')"
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </div>
      </header>

      <div v-if="isMobile" class="evidence-mobile-tabs" role="tablist" aria-label="照片、线索与参考">
        <button type="button" role="tab" :aria-selected="mobileTab === 'photo'" :class="{ active: mobileTab === 'photo' }" @click="emit('open-tab', 'photo')">照片</button>
        <button type="button" role="tab" :aria-selected="mobileTab === 'clues'" :class="{ active: mobileTab === 'clues' }" @click="emit('open-tab', 'clues')">线索 <span>{{ clueCount }}</span></button>
        <button type="button" role="tab" :aria-selected="mobileTab === 'references'" :class="{ active: mobileTab === 'references' }" @click="emit('open-tab', 'references')">参考</button>
      </div>

      <div class="evidence-scroll">
        <section v-if="!isMobile || mobileTab === 'photo'" class="evidence-section evidence-section--photo">
          <div class="dock-section-heading">
            <div>
              <span class="section-kicker">PHOTO</span>
              <h3>照片</h3>
            </div>
            <span v-if="imagePresent" class="dock-muted">点击更换</span>
          </div>

          <UploadPanel
            :preview-url="previewUrl"
            :click-points="clickPoints"
            :disabled="uploading"
            @file="emit('file', $event)"
          />
          <div v-if="uploadNotice" class="inline-error">{{ uploadNotice }}</div>
          <div v-if="selectedFileName" class="file-chip">
            <span>{{ selectedFileName }}</span>
            <span class="state">{{ hasSession && !hasLocalPreview ? '已分析' : '待上传' }}</span>
          </div>
          <button
            v-if="selectedFileName && (!hasSession || hasLocalPreview)"
            type="button"
            class="primary-btn upload-btn"
            :disabled="uploading"
            @click="emit('upload')"
          >
            {{ uploading ? '上传并分析中…' : '上传并分析' }}
          </button>

          <div class="evidence-actions">
            <button
              type="button"
              class="ghost-btn evidence-viewer-btn"
              :disabled="!imagePresent"
              @click="emit('open-viewer')"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
                <circle cx="11" cy="11" r="6.5" />
                <path d="m16 16 4 4M8 11h6M11 8v6" />
              </svg>
              区域分析
            </button>
            <span class="evidence-footnote">分析结果仅作参考 · 位置不是照片 GPS</span>
          </div>
        </section>

        <section v-if="!isMobile || mobileTab === 'clues'" class="evidence-section evidence-section--clues">
          <div class="dock-section-heading">
            <div>
              <span class="section-kicker">OBSERVATIONS</span>
              <h3>线索 <span class="dock-count">{{ clueCount }}</span></h3>
            </div>
            <span class="dock-muted">按可靠度</span>
          </div>
          <ClueList
            :clues="clues"
            :loading="uploading"
            :recent-clue-keys="recentClueKeys"
            @locate-clue="emit('locate-clue', $event)"
          />
        </section>

        <section v-if="!isMobile || mobileTab === 'references'" class="evidence-section evidence-section--references">
          <div class="dock-section-heading">
            <div>
              <span class="section-kicker">REFERENCES</span>
              <h3>参考街景</h3>
            </div>
            <span class="dock-muted">仅供参考</span>
          </div>
          <CandidatePanel
            v-bind="candidates"
            surface="references"
            embedded
            :guided-mode="guidedActive"
            :guided-country-allowed="guidedCountryAllowed"
          />
        </section>
      </div>
    </section>
  </aside>
</template>

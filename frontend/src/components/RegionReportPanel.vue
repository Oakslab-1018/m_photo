<script setup lang="ts">
import { computed } from 'vue'
import type {
  RegionAnalysisRecord,
  AnalyzePointRequest,
  ObservationAdoptionRequest,
  ObservationAdoptionSubmission,
  RegionObservation,
  RegionReport,
  RegionQuestion,
} from '@/types'
import { observationBox } from '@/utils/regionAnalysis'

const props = withDefaults(
  defineProps<{
    imageUrl?: string | null
    reports?: RegionReport[]
    selectedReportId?: string | null
    requests?: RegionAnalysisRecord[]
    analyzing?: boolean
    disabled?: boolean
    observationAdoption?: ObservationAdoptionSubmission | null
    observationAdoptionRetryable?: boolean
  }>(),
  {
    imageUrl: null,
    reports: () => [],
    selectedReportId: null,
    requests: () => [],
    analyzing: false,
    disabled: false,
    observationAdoption: null,
    observationAdoptionRetryable: false,
  },
)

const emit = defineEmits<{
  'select-region': [reportId: string]
  'retry-region': [request: AnalyzePointRequest]
  'ask-question': [question: RegionQuestion]
  'adopt-observation': [request: ObservationAdoptionRequest]
  'retry-observation-adoption': []
}>()

const selectedReport = computed(() =>
  props.reports.find((report) => report.id === props.selectedReportId) ?? props.reports[0] ?? null,
)

const waitReason = computed(() => props.disabled || props.analyzing)

function purposeLabel(purpose: RegionReport['purpose']): string {
  if (purpose === 'ocr') return '文字'
  if (purpose === 'scene') return '场景'
  return '自动'
}

function dedupeLabel(report: RegionReport): string {
  switch (report.dedupe) {
    case 'new':
      return '新增证据'
    case 'duplicate_region':
      return '相同区域，已有报告'
    case 'duplicate_request':
      return '重复请求，已复用报告'
    case 'no_new_evidence':
      return '未发现新增证据'
  }
}

function kindLabel(kind: RegionObservation['kind']): string {
  switch (kind) {
    case 'ocr_text':
      return '识别文字'
    case 'semantic':
      return '语义线索'
    case 'pixel_signal':
      return '像素测量'
    case 'scene_hypothesis':
      return '场景假设（复核参考）'
    case 'crop_ranking':
      return '裁剪排序证据'
  }
}

function trustLabel(observation: RegionObservation): string {
  const trust = observation.trust
  const level = trust.level === 'high' ? '高可信' : trust.level === 'medium' ? '中等可信' : '低可信'
  const flags: string[] = [level]
  if (trust.reliable) flags.push('可靠')
  if (trust.validated) flags.push('已核对')
  if (trust.affects_ranking) flags.push('影响排序')
  if (trust.uncertain) flags.push('不确定')
  return flags.join(' · ')
}

function providerState(status: { enabled: boolean; available: boolean; state: string }): string {
  if (!status.enabled) return '未启用'
  if (!status.available) return '不可用'
  const state = status.state.toLowerCase()
  if (!state || state === 'unknown') return '可用'
  if (state === 'ok' || state === 'ready' || state === 'available' || state === 'healthy') return '可用'
  if (state === 'mock') return '模拟模式（仅开发，不可靠）'
  return status.state
}

function formatNumber(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

const visibleObservations = computed(() => {
  const report = selectedReport.value
  if (!report) return []
  return report.observations.filter((observation) => {
    // 像素统计只是测量信号，不是分析依据，不在报告中展示
    if (observation.kind === 'pixel_signal') return false
    if (observation.kind !== 'ocr_text') return true
    return canAdoptObservation(report, observation)
  })
})

function cropPreviewStyle(report: RegionReport | null): Record<string, string> {
  if (!report) return {}
  const bbox = report.bbox_norm
  const width = report.image.width * bbox.w
  const height = report.image.height * bbox.h
  return width > 0 && height > 0 ? { aspectRatio: String(width / height) } : {}
}

function cropImageStyle(report: RegionReport | null): Record<string, string> {
  if (!report) return {}
  const bbox = report.bbox_norm
  return {
    width: String(100 / bbox.w) + '%',
    height: String(100 / bbox.h) + '%',
    left: String(-(bbox.x / bbox.w) * 100) + '%',
    top: String(-(bbox.y / bbox.h) * 100) + '%',
  }
}

function bboxText(report: RegionReport | null): string {
  if (!report) return '—'
  if (report.bbox_px.length === 4) {
    return '像素框 ' + report.bbox_px.join(' / ')
  }
  return '归一化框 ' + formatNumber(report.bbox_norm.x, 3) + ', ' + formatNumber(report.bbox_norm.y, 3)
}

function observationHasBox(observation: RegionObservation): boolean {
  return observationBox(observation) !== null
}

/** Only real, available, non-mock OCR text can become an adoption request. */
function canAdoptObservation(report: RegionReport, observation: RegionObservation): boolean {
  return (
    observation.kind === 'ocr_text' &&
    report.ocr.available &&
    report.ocr.state.toLowerCase() !== 'mock' &&
    Boolean(observation.text?.trim())
  )
}

function adoptionFor(
  report: RegionReport,
  observation: RegionObservation,
): ObservationAdoptionSubmission | null {
  const adoption = props.observationAdoption
  if (!adoption) return null
  return adoption.report_id === report.id && adoption.observation_id === observation.id
    ? adoption
    : null
}

function adoptionSettled(report: RegionReport, observation: RegionObservation): boolean {
  const status = adoptionFor(report, observation)?.status
  return status === 'applied' || status === 'unchanged'
}

function adoptionStatusText(adoption: ObservationAdoptionSubmission): string {
  const summary = adoption.receipt?.summary?.trim()
  if (adoption.status === 'pending') {
    return '正在确认 OCR 原文并检索地点；不会自动确认地点。'
  }
  if (adoption.status === 'applied') {
    return summary
      ? `已确认文字；助手反馈：${summary}`
      : '已确认 OCR 原文并提交地点检索；不会自动确认地点。'
  }
  if (adoption.status === 'unchanged') {
    return summary
      ? `这段 OCR 原文此前已确认；助手反馈：${summary}`
      : '这段 OCR 原文此前已确认，无需重复提交。'
  }
  if (adoption.status === 'unconfirmed') {
    return adoption.error?.trim() || '服务器未返回可识别的采用回执，请核对最新引导状态后再试。'
  }
  return adoption.error?.trim() || '确认文字失败，地点未被自动确认。'
}

function adoptionIsPending(report: RegionReport, observation: RegionObservation): boolean {
  return adoptionFor(report, observation)?.status === 'pending'
}

function adoptionNeedsRetry(report: RegionReport, observation: RegionObservation): boolean {
  return (
    props.observationAdoptionRetryable &&
    adoptionFor(report, observation)?.status === 'failed'
  )
}

function adoptionTextFor(report: RegionReport, observation: RegionObservation): string {
  const adoption = adoptionFor(report, observation)
  return adoption ? adoptionStatusText(adoption) : ''
}

function adoptionClassFor(report: RegionReport, observation: RegionObservation): string {
  const status = adoptionFor(report, observation)?.status
  return status ? `is-${status}` : ''
}
</script>

<template>
  <aside class="region-report-panel" aria-label="局部区域分析报告">
    <div class="region-report-head">
      <div>
        <span class="region-report-kicker">LOCAL ANALYSIS</span>
        <h3>局部区域报告</h3>
      </div>
      <span class="region-report-count">{{ reports.length }} 份</span>
    </div>

    <div v-if="requests.length" class="region-request-list">
      <div v-for="record in requests" :key="record.id" class="region-request-row" :class="record.status">
        <span class="region-request-state">{{ record.status === 'pending' ? '分析中…' : '分析失败' }}</span>
        <span class="region-request-id">{{ record.id }}</span>
        <p v-if="record.error" class="region-request-error">{{ record.error }}</p>
        <button
          v-if="record.status === 'failed'"
          type="button"
          class="ghost-btn region-retry-btn"
          :disabled="waitReason"
          @click="emit('retry-region', record.request)"
        >
          使用同一请求 ID 重试
        </button>
      </div>
    </div>

    <div v-if="!reports.length" class="region-report-empty">
      尚未收到区域报告。旧版响应只显示原有线索，不会伪造 OCR 或模型结论。
    </div>

    <template v-else>
      <div class="region-report-history" aria-label="区域报告历史">
        <button
          v-for="report in reports"
          :key="report.id"
          type="button"
          class="region-history-item"
          :class="{ selected: report.id === selectedReport?.id }"
          @click="emit('select-region', report.id)"
        >
          <span class="region-history-purpose">{{ purposeLabel(report.purpose) }}</span>
          <strong>{{ report.summary || '局部区域分析' }}</strong>
          <small>{{ dedupeLabel(report) }} · {{ report.id }}</small>
        </button>
      </div>

      <article v-if="selectedReport" class="region-report-detail">
        <p class="region-report-summary">{{ selectedReport.summary || '服务器未提供摘要。' }}</p>

        <div class="region-crop-preview" :style="cropPreviewStyle(selectedReport)">
          <img
            v-if="imageUrl"
            :src="imageUrl"
            alt="区域裁剪预览"
            :style="cropImageStyle(selectedReport)"
          >
          <span v-else>原图不可用</span>
        </div>

        <div class="region-meta-grid">
          <span>用途：{{ purposeLabel(selectedReport.purpose) }}</span>
          <span>尺度：{{ formatNumber(selectedReport.scale, 2) }}</span>
          <span>{{ bboxText(selectedReport) }}</span>
          <span>EXIF 方向：{{ selectedReport.image.orientation_applied ? '已应用' : '无需处理' }}</span>
        </div>

        <section class="region-provider-grid">
          <div class="region-provider-card" :class="{ unavailable: !selectedReport.ocr.available }">
            <span class="region-provider-title">OCR · {{ selectedReport.ocr.provider }}</span>
            <strong>{{ providerState(selectedReport.ocr) }}</strong>
            <small>识别行数 {{ selectedReport.ocr.line_count }}</small>
          </div>
          <div class="region-provider-card" :class="{ unavailable: !selectedReport.model.available }">
            <span class="region-provider-title">模型 · {{ selectedReport.model.provider }}</span>
            <strong>{{ providerState(selectedReport.model) }}</strong>
            <small>
              {{ selectedReport.model.ranking_applied ? '已参与弱排序' : '未参与排序' }}
              <template v-if="selectedReport.model.scene_available === false"> · 场景端点不可用</template>
            </small>
            <small v-if="typeof selectedReport.model.margin === 'number'">
              排序诊断边界差 {{ formatNumber(selectedReport.model.margin, 3) }}（非概率）
            </small>
            <small v-if="selectedReport.model.uncertain">模型标记为不确定，仅作复核参考</small>
            <small v-if="selectedReport.model.ranking_note">{{ selectedReport.model.ranking_note }}</small>
          </div>
        </section>

        <section class="region-observations">
          <div class="region-section-title">分析依据</div>
          <div v-if="!visibleObservations.length" class="region-report-empty">
            没有可展示的真实 OCR/模型观察；不可用状态不会被猜测文本填充。
          </div>
          <article v-for="observation in visibleObservations" :key="observation.id" class="region-observation-card">
            <div class="region-observation-head">
              <span class="region-observation-kind">{{ kindLabel(observation.kind) }}</span>
              <strong>{{ observation.label }}</strong>
            </div>
            <p v-if="observation.kind === 'ocr_text' && observation.text" class="region-ocr-text">
              {{ observation.text }}
            </p>
            <p class="region-observation-evidence">{{ observation.evidence }}</p>
            <div class="region-observation-meta">
              <span>{{ observation.source }}{{ observation.provider ? ' · ' + observation.provider : '' }}</span>
              <span>{{ trustLabel(observation) }}</span>
              <span v-if="!observation.trust.affects_ranking">仅供参考，不影响地理排序</span>
              <span v-if="typeof observation.trust.confidence === 'number'">
                {{ observation.trust.confidence_kind || '置信度' }} {{ formatNumber(observation.trust.confidence, 3) }}
              </span>
              <span v-if="typeof observation.trust.raw_cosine === 'number'">
                参考余弦 {{ formatNumber(observation.trust.raw_cosine, 3) }}（非概率）
              </span>
              <span v-if="observationHasBox(observation)">图中已标出证据框</span>
            </div>
            <div
              v-if="canAdoptObservation(selectedReport, observation)"
              class="region-adoption"
            >
              <div class="region-adoption-actions">
                <button
                  v-if="!adoptionSettled(selectedReport, observation) && !adoptionNeedsRetry(selectedReport, observation)"
                  type="button"
                  class="primary-btn region-adopt-btn"
                  :disabled="waitReason || adoptionIsPending(selectedReport, observation)"
                  @click="emit('adopt-observation', { report_id: selectedReport.id, observation_id: observation.id })"
                >
                  {{
                    adoptionIsPending(selectedReport, observation)
                      ? '正在确认文字…'
                      : adoptionFor(selectedReport, observation)?.status === 'failed'
                        ? '重新确认文字并用于定位'
                        : '确认文字并用于定位'
                  }}
                </button>
                <button
                  v-if="adoptionNeedsRetry(selectedReport, observation)"
                  type="button"
                  class="ghost-btn region-adopt-retry"
                  :disabled="waitReason"
                  @click="emit('retry-observation-adoption')"
                >
                  重试上次采用（同一事件）
                </button>
              </div>
              <p
                v-if="adoptionFor(selectedReport, observation)"
                class="region-adoption-status"
                :class="adoptionClassFor(selectedReport, observation)"
                :role="adoptionFor(selectedReport, observation)?.status === 'failed' ? 'alert' : 'status'"
              >
                {{ adoptionTextFor(selectedReport, observation) }}
              </p>
              <p class="region-adoption-hint">
                仅提交报告 ID 与观察 ID；后端使用已保存的 OCR 原文检索，不会自动确认地点。
              </p>
            </div>
            <p v-if="observation.kind === 'scene_hypothesis'" class="region-scene-note">
              场景假设仅用于人工复核，不是拍摄地概率，也不参与引导确认。
            </p>
          </article>
        </section>

        <section v-if="selectedReport.question_suggestions.length" class="region-questions">
          <div class="region-section-title">建议追问（仅填入草稿）</div>
          <button
            v-for="question in selectedReport.question_suggestions"
            :key="question.id"
            type="button"
            class="region-question-btn"
            :disabled="waitReason"
            @click="emit('ask-question', question)"
          >
            <strong>{{ question.text }}</strong>
            <small>{{ question.reason }} · 依据 {{ question.basis_observation_ids.length }} 条观察 · 点击只填入追问草稿，不确认文字，也不会自动确认地点</small>
          </button>
        </section>
      </article>
    </template>
  </aside>
</template>

<style scoped>
.region-adoption {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 8px;
  border-top: 1px dashed var(--line-faint);
}
.region-adoption-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.region-adopt-btn {
  min-height: 34px;
  padding: 6px 10px;
  font-size: 13px;
}
.region-adopt-retry {
  min-height: 34px;
  padding: 6px 10px;
  font-size: 13px;
}
.region-adoption-status {
  color: var(--muted);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
}
.region-adoption-status.is-pending {
  color: var(--amber);
}
.region-adoption-status.is-applied,
.region-adoption-status.is-unchanged {
  color: var(--sage);
}
.region-adoption-status.is-unconfirmed {
  color: var(--warn);
}
.region-adoption-status.is-failed {
  color: var(--coral);
}
.region-adoption-hint {
  color: var(--faint);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
}
</style>

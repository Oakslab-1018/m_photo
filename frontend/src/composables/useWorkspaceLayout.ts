import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'
import { useImmersiveLayout } from '@/composables/useImmersiveLayout'

export type FocusMode = 'collab' | 'map' | 'photo'
export type MobileSheetState = 'peek' | 'half' | 'full'
export type EvidenceTab = 'photo' | 'clues' | 'references'
export type WorkspaceModal = 'history' | 'evidence'

export interface UseWorkspaceLayoutOptions {
  /** Reactive business gate owned by the caller. */
  historyAllowed: () => boolean
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[contenteditable="true"]',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

/**
 * Own responsive workspace presentation and modal focus management. Selection
 * and gating remain caller-owned business concerns; this composable only reads
 * the latter through a narrow reactive getter.
 */
export function useWorkspaceLayout(options: UseWorkspaceLayoutOptions) {
  const commandBarRef = ref<HTMLElement | null>(null)
  const mapHostRef = ref<HTMLElement | null>(null)
  const evidenceDockRef = ref<HTMLElement | null>(null)
  const reasoningDockRef = ref<HTMLElement | null>(null)
  const focusMode = ref<FocusMode>('collab')
  const leftCollapsed = ref(false)
  const rightCollapsed = ref(false)
  const reasoningWide = ref(false)
  const mobileSheetState = ref<MobileSheetState>('half')
  const mobileEvidenceOpen = ref(false)
  const mobileEvidenceTab = ref<EvidenceTab>('photo')
  const historyOpen = ref(false)
  const historyDialogRef = ref<HTMLElement | null>(null)
  const mobileEvidenceDialogRef = ref<HTMLElement | null>(null)

  const { mapInsets, isMobile, isCompact } = useImmersiveLayout({
    commandBar: commandBarRef,
    mapHost: mapHostRef,
    leftDock: evidenceDockRef,
    rightDock: reasoningDockRef,
    includeTop: () => true,
    includeLeft: () => focusMode.value !== 'map',
    includeRight: () => focusMode.value !== 'map',
    includeBottom: () => !mobileEvidenceOpen.value,
  })

  const historyAllowed = computed(() => options.historyAllowed())
  const mobileEvidenceModalOpen = computed(
    () => isMobile.value && mobileEvidenceOpen.value,
  )
  const backgroundInert = computed(
    () => historyOpen.value || mobileEvidenceModalOpen.value,
  )
  const activeModal = computed<WorkspaceModal | null>(() => {
    if (historyOpen.value) return 'history'
    if (mobileEvidenceModalOpen.value) return 'evidence'
    return null
  })

  let modalReturnFocus: HTMLElement | null = null
  let skipModalFocusRestore = false
  let modalFocusEpoch = 0

  function modalElement(modal: WorkspaceModal): HTMLElement | null {
    return modal === 'history' ? historyDialogRef.value : mobileEvidenceDialogRef.value
  }

  function focusableElements(container: HTMLElement): HTMLElement[] {
    return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
      (element) => element.getClientRects().length > 0 && element.getAttribute('aria-hidden') !== 'true',
    )
  }

  function focusActiveModal(modal: WorkspaceModal): void {
    const container = modalElement(modal)
    if (!container) return
    const preferred = container.querySelector<HTMLElement>('[data-modal-autofocus]')
    const target = preferred ?? focusableElements(container)[0] ?? container
    target.focus()
  }

  function trapModalFocus(event: KeyboardEvent, modal: WorkspaceModal): void {
    const container = modalElement(modal)
    if (!container) return
    const focusable = focusableElements(container)
    if (!focusable.length) {
      event.preventDefault()
      container.focus()
      return
    }
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    const active = document.activeElement
    if (event.shiftKey && (active === first || active === container || !container.contains(active))) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && (active === last || active === container || !container.contains(active))) {
      event.preventDefault()
      first.focus()
    }
  }

  function restoreModalFocus(target: HTMLElement | null): void {
    if (!target?.isConnected || target.getClientRects().length === 0) return
    target.focus()
  }

  const evidenceAnchorVisible = computed(
    () => (isMobile.value ? mobileEvidenceOpen.value : focusMode.value !== 'map'),
  )
  const reasoningAnchorVisible = computed(
    () => (isMobile.value ? !mobileEvidenceOpen.value : focusMode.value !== 'map'),
  )
  const showEvidenceThumbnail = computed(() => !isMobile.value && leftCollapsed.value)
  const showReasoningTaskChip = computed(() => !isMobile.value && rightCollapsed.value)

  function setFocusMode(mode: FocusMode): void {
    focusMode.value = mode
    historyOpen.value = false
    if (isMobile.value) {
      if (mode === 'photo') {
        mobileEvidenceOpen.value = true
        mobileEvidenceTab.value = 'photo'
        mobileSheetState.value = 'peek'
      } else if (mode === 'map') {
        mobileEvidenceOpen.value = false
        mobileSheetState.value = 'peek'
      } else {
        mobileEvidenceOpen.value = false
        mobileSheetState.value = 'half'
      }
      return
    }
    if (mode === 'photo') {
      leftCollapsed.value = false
      rightCollapsed.value = true
    } else if (mode === 'map') {
      leftCollapsed.value = true
      rightCollapsed.value = true
    } else {
      rightCollapsed.value = false
      if (!isCompact.value) leftCollapsed.value = false
    }
  }

  function expandEvidenceDock(): void {
    leftCollapsed.value = false
    if (focusMode.value === 'map') focusMode.value = 'collab'
  }

  function collapseEvidenceDock(): void {
    leftCollapsed.value = true
  }

  function expandReasoningDock(): void {
    rightCollapsed.value = false
    if (focusMode.value === 'map') focusMode.value = 'collab'
  }

  function collapseReasoningDock(): void {
    rightCollapsed.value = true
  }

  function toggleReasoningWidth(): void {
    reasoningWide.value = !reasoningWide.value
  }

  function setMobileSheetState(state: MobileSheetState): void {
    mobileSheetState.value = state
  }

  function openMobileEvidence(tab: EvidenceTab): void {
    historyOpen.value = false
    mobileEvidenceTab.value = tab
    mobileEvidenceOpen.value = true
    mobileSheetState.value = 'peek'
  }

  function closeMobileEvidence(): void {
    mobileEvidenceOpen.value = false
  }

  /**
   * Preserve the viewer handoff: close the mobile evidence dialog without
   * returning focus to a dock that is about to be covered by the viewer.
   */
  function closeMobileEvidenceForViewer(): void {
    if (mobileEvidenceOpen.value) skipModalFocusRestore = true
    mobileEvidenceOpen.value = false
  }

  function openHistory(): void {
    if (!historyAllowed.value) return
    mobileEvidenceOpen.value = false
    historyOpen.value = true
  }

  function closeHistory(): void {
    historyOpen.value = false
  }

  function restoreDefaultDocks(): void {
    rightCollapsed.value = false
    if (!isCompact.value) leftCollapsed.value = false
  }

  function onGlobalKeydown(event: KeyboardEvent): void {
    const modal = activeModal.value
    if (!modal) return
    if (event.key === 'Tab') {
      trapModalFocus(event, modal)
      return
    }
    if (event.key !== 'Escape') return
    event.preventDefault()
    event.stopPropagation()
    if (modal === 'history') closeHistory()
    else closeMobileEvidence()
  }

  watch(
    isCompact,
    (compact) => {
      if (compact && !isMobile.value) leftCollapsed.value = true
    },
    { immediate: true },
  )

  watch(isMobile, (mobile) => {
    historyOpen.value = false
    mobileEvidenceOpen.value = false
    if (mobile) mobileSheetState.value = 'half'
  })

  watch(historyAllowed, (allowed) => {
    if (!allowed) historyOpen.value = false
  })

  watch(
    activeModal,
    async (next, previous) => {
      const epoch = ++modalFocusEpoch
      if (next) {
        if (!previous) {
          modalReturnFocus = document.activeElement instanceof HTMLElement
            ? document.activeElement
            : null
        }
        await nextTick()
        if (epoch === modalFocusEpoch) focusActiveModal(next)
        return
      }
      if (!previous) return

      const restoreTarget = modalReturnFocus
      modalReturnFocus = null
      await nextTick()
      if (epoch !== modalFocusEpoch) return
      if (skipModalFocusRestore) {
        skipModalFocusRestore = false
        return
      }
      restoreModalFocus(restoreTarget)
    },
  )

  onMounted(() => {
    window.addEventListener('keydown', onGlobalKeydown)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('keydown', onGlobalKeydown)
  })

  return {
    commandBarRef,
    mapHostRef,
    evidenceDockRef,
    reasoningDockRef,
    historyDialogRef,
    mobileEvidenceDialogRef,
    focusMode,
    leftCollapsed,
    rightCollapsed,
    reasoningWide,
    mobileSheetState,
    mobileEvidenceOpen,
    mobileEvidenceTab,
    historyOpen,
    mapInsets,
    isMobile,
    isCompact,
    historyAllowed,
    mobileEvidenceModalOpen,
    backgroundInert,
    activeModal,
    evidenceAnchorVisible,
    reasoningAnchorVisible,
    showEvidenceThumbnail,
    showReasoningTaskChip,
    setFocusMode,
    expandEvidenceDock,
    collapseEvidenceDock,
    expandReasoningDock,
    collapseReasoningDock,
    toggleReasoningWidth,
    setMobileSheetState,
    openMobileEvidence,
    closeMobileEvidence,
    closeMobileEvidenceForViewer,
    openHistory,
    closeHistory,
    restoreDefaultDocks,
  }
}


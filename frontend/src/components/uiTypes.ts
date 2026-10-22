import type {
  CityScore,
  CountryScore,
  GuidedDialogue,
  GuidedEventReceipt,
  GuidedPhase,
  GuidedStage,
  HypothesisAction,
  ConfidenceRoute,
  LocationCandidate,
} from '@/types'
import type { GuidedFreeTextSubmission } from '@/composables/useGuidedComposer'
import type { FeedbackReceiptTone } from '@/utils/feedback'
import type { CountryConfidenceShortcut } from '@/utils/guided'

export type {
  EvidenceTab,
  FocusMode,
  MobileSheetState,
} from '@/composables/useWorkspaceLayout'

export interface LiveStatus {
  tone: FeedbackReceiptTone | 'info'
  text: string
}

/** Shared data contract for the candidate/reference surface. */
export interface CandidatePanelBindings {
  scores: CountryScore[]
  confirmedIso: string | null
  selectedIso: string | null
  exclusions: Array<{ iso: string; label: string }>
  busy: boolean
  guidedCandidates: LocationCandidate[]
  guidedCandidatesUnlocked: boolean
  guidedConfirmationReady: boolean
  guidedDataReady: boolean
  guidedStage: GuidedStage | null
  cityScores: CityScore[]
  cityCountryIso: string | null
  citiesStatus: string | null
  countryOptions: Array<{ iso: string; name_zh: string }>
  cityBusy: boolean
  cityNotice: string | null
  citySourceIso: string | null
  citySourceLabel: string | null
}

/** Shared data contract for the guided conversation panel. */
export interface GuidedDialogueBindings {
  dialogue: GuidedDialogue | null
  candidates: LocationCandidate[]
  action: HypothesisAction | null
  confidence: ConfidenceRoute | null
  confidenceShortcut: CountryConfidenceShortcut | null
  confidenceCandidates: LocationCandidate[]
  loading: boolean
  applying: boolean
  externalBusy: boolean
  error: string | null
  conflict: string | null
  canRetry: boolean
  receipt: GuidedEventReceipt | null
  selectedGuidedTargetKey: string | null
  suggestedFreeText: string | null
  freeTextSubmission: GuidedFreeTextSubmission | null
  phase: GuidedPhase
  candidatesUnlocked: boolean
  canReviewBack: boolean
  locationConfirmationReady: boolean
  recentClueIds: string[]
}

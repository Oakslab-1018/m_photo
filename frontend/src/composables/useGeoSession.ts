import { computed, ref } from 'vue'
import type {
  AnalyzePointResponse,
  ChatMessage,
  Clue,
  CountryScore,
  FeedbackResponse,
  HypothesisAction,
  UploadResponse,
} from '@/types'
import { sortScores } from '@/utils/map'

/** Central session state shared by App and presentational components. */
export function useGeoSession() {
  const sessionId = ref<number | null>(null)
  const imageUrl = ref<string | null>(null)
  const clues = ref<Clue[]>([])
  const scores = ref<CountryScore[]>([])
  const messages = ref<ChatMessage[]>([])
  const action = ref<HypothesisAction | null>(null)
  const exclusions = ref<string[]>([])
  const confirmedIso = ref<string | null>(null)
  const done = ref(false)

  const topCandidate = computed<CountryScore | null>(() => scores.value[0] ?? null)
  const isConfirmed = computed(() => confirmedIso.value !== null)
  const acceptedCountry = computed<CountryScore | null>(() => {
    if (!confirmedIso.value) return null
    return scores.value.find((item) => item.iso === confirmedIso.value) ?? null
  })

  function applyUpload(response: UploadResponse): void {
    sessionId.value = response.session_id
    imageUrl.value = response.image_url
    clues.value = [...(response.clues ?? [])]
    scores.value = sortScores(response.scores ?? [])
    messages.value = [...(response.messages ?? [])]
    action.value = response.action ?? null
    exclusions.value = []
    confirmedIso.value = null
    done.value = false
  }

  function applyFeedback(response: FeedbackResponse): void {
    // Confirm ("done") responses may carry `scores: []`. Handle done first so
    // an empty list never wipes the candidate map / confirmed marker.
    if (response.done) {
      if (response.clues) clues.value = [...response.clues]
      if (response.messages) messages.value = [...response.messages]
      done.value = true
      confirmedIso.value = response.iso ?? confirmedIso.value
      action.value = { kind: 'done', payload: response.iso ? { iso: response.iso } : {} }
      return
    }

    if (response.clues) clues.value = [...response.clues]
    if (response.scores) scores.value = sortScores(response.scores)
    if (response.messages) messages.value = [...response.messages]
    if (response.exclusions) exclusions.value = [...response.exclusions]
    if (response.action) action.value = response.action
  }

  function applyPointAnalysis(response: AnalyzePointResponse): void {
    if (response.clues) clues.value = [...response.clues]
    if (response.scores) scores.value = sortScores(response.scores)
    if (response.messages) messages.value = [...response.messages]
    if (response.exclusions) exclusions.value = [...response.exclusions]
    if (response.action) action.value = response.action
  }

  function reset(): void {
    sessionId.value = null
    imageUrl.value = null
    clues.value = []
    scores.value = []
    messages.value = []
    action.value = null
    exclusions.value = []
    confirmedIso.value = null
    done.value = false
  }

  return {
    sessionId,
    imageUrl,
    clues,
    scores,
    messages,
    action,
    exclusions,
    confirmedIso,
    done,
    topCandidate,
    isConfirmed,
    acceptedCountry,
    applyUpload,
    applyFeedback,
    applyPointAnalysis,
    reset,
  }
}

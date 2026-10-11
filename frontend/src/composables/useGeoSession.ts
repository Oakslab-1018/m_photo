import { computed, ref } from 'vue'
import type {
  AnalyzePointResponse,
  ChatMessage,
  CitiesResponse,
  CityScore,
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
  const cityScores = ref<CityScore[]>([])
  const cityCountryIso = ref<string | null>(null)
  const citiesStatus = ref<CitiesResponse['status'] | null>(null)

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
    // Optional upload payload may already carry city candidates; absent stays empty.
    cityScores.value = [...(response.city_scores ?? [])]
    cityCountryIso.value = null
    citiesStatus.value = response.city_scores ? 'ok' : null
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
    // A missing action means no pending question: never keep stale answer buttons.
    action.value = response.action ?? null
  }

  /** Apply a cities response; empty results clear any previous city candidates. */
  function applyCities(response: CitiesResponse): void {
    cityScores.value = [...(response.city_scores ?? [])]
    cityCountryIso.value = response.country_iso ?? null
    citiesStatus.value = response.status ?? null
  }

  function clearCities(): void {
    cityScores.value = []
    cityCountryIso.value = null
    citiesStatus.value = null
  }

  function applyPointAnalysis(response: AnalyzePointResponse): void {
    if (response.clues) clues.value = [...response.clues]
    if (response.scores) scores.value = sortScores(response.scores)
    if (response.messages) messages.value = [...response.messages]
    if (response.exclusions) exclusions.value = [...response.exclusions]
    // Point analysis replaces the pending question, or clears it when absent.
    action.value = response.action ?? null
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
    cityScores.value = []
    cityCountryIso.value = null
    citiesStatus.value = null
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
    cityScores,
    cityCountryIso,
    citiesStatus,
    topCandidate,
    isConfirmed,
    acceptedCountry,
    applyUpload,
    applyFeedback,
    applyCities,
    clearCities,
    applyPointAnalysis,
    reset,
  }
}

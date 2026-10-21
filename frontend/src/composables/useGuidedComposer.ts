import { computed, ref, watch, type ComputedRef, type Ref } from 'vue'

export type GuidedFreeTextSubmissionStatus =
  | 'pending'
  | 'applied'
  | 'unchanged'
  | 'unconfirmed'
  | 'failed'

/** Parent-owned acknowledgement for one guided conversation submission. */
export interface GuidedFreeTextSubmission {
  text: string
  status: GuidedFreeTextSubmissionStatus
  error?: string | null
}

export interface UseGuidedComposerOptions {
  suggestedText: () => string | null | undefined
  submission: () => GuidedFreeTextSubmission | null | undefined
  isBusy: () => boolean
  onSend: (text: string) => void
}

export interface UseGuidedComposer {
  text: Ref<string>
  composing: Ref<boolean>
  canSend: ComputedRef<boolean>
  submit: () => boolean
  onEnter: (event: KeyboardEvent) => void
}

function normalized(value: string | null | undefined): string {
  return value?.trim() ?? ''
}

/**
 * Shared draft/submission state for the guided conversation composer.
 *
 * The draft is cleared only after a positive applied/unchanged acknowledgement
 * for the same text. Pending, failed, stale/conflicted and mismatched
 * acknowledgements keep whatever the user has typed, so a server retry never
 * silently loses a clue.
 */
export function useGuidedComposer(options: UseGuidedComposerOptions): UseGuidedComposer {
  const text = ref('')
  const composing = ref(false)
  const canSend = computed(
    () => !options.isBusy() && !composing.value && normalized(text.value).length > 0,
  )

  watch(
    () => normalized(options.suggestedText()),
    (suggested) => {
      // Suggestions can prepopulate the composer, but an empty suggestion is
      // never proof that a submitted draft should be discarded.
      if (suggested) text.value = suggested
    },
    { immediate: true },
  )

  watch(
    () => options.submission(),
    (submission) => {
      if (!submission) return
      const submitted = normalized(submission.text)
      if (!submitted) return
      if (
        (submission.status === 'applied' || submission.status === 'unchanged') &&
        normalized(text.value) === submitted
      ) {
        text.value = ''
        return
      }
      // A failed or stale submission restores its text only when the composer
      // is empty; any newer user edit wins.
      if (
        (submission.status === 'failed' || submission.status === 'unconfirmed') &&
        !normalized(text.value)
      ) {
        text.value = submitted
      }
    },
  )

  function submit(): boolean {
    if (options.isBusy() || composing.value) return false
    const value = normalized(text.value)
    if (!value) return false
    options.onSend(value)
    return true
  }

  function onEnter(event: KeyboardEvent): void {
    // Enter confirms an active IME composition; it must never send the clue.
    if (event.isComposing || composing.value) return
    event.preventDefault()
    submit()
  }

  return { text, composing, canSend, submit, onEnter }
}

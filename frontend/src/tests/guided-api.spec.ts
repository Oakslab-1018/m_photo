import { describe, expect, it, vi } from 'vitest'
import { GeoApi } from '@/api/geo'
import type { GuidedDialogueResponse, GuidedEventPayload } from '@/types'

const responseBody: GuidedDialogueResponse = {
  dialogue: {
    version: 1,
    revision: 1,
    stage: 'country',
    confirmed: { country: null, admin1: null, city: null },
    exclusions: [],
    clues: [],
    pending_question: null,
  },
  candidates: [],
  action: null,
  messages: [],
  scores: [],
  clues: [],
  exclusions: [],
}

describe('GeoApi guided dialogue routes', () => {
  it('calls the start, snapshot and event endpoints with the contract payload', async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify(responseBody), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new GeoApi({ fetchImpl })
    const event: GuidedEventPayload = {
      type: 'free_text',
      event_id: 'event-1',
      expected_revision: 1,
      text: '北海道',
    }

    await client.dialogueStart(9)
    await client.dialogue(9)
    await client.dialogueEvent(9, event)

    expect(fetchImpl.mock.calls[0]?.[0]).toBe('/api/geo/9/dialogue/start')
    expect(fetchImpl.mock.calls[0]?.[1]).toMatchObject({ method: 'POST' })
    expect(fetchImpl.mock.calls[1]?.[0]).toBe('/api/geo/9/dialogue')
    expect(fetchImpl.mock.calls[1]?.[1]).toBeUndefined()
    expect(fetchImpl.mock.calls[2]?.[0]).toBe('/api/geo/9/dialogue/events')
    expect(fetchImpl.mock.calls[2]?.[1]).toMatchObject({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })
    expect(JSON.parse(String(fetchImpl.mock.calls[2]?.[1]?.body))).toEqual(event)
  })
})

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

describe('GeoApi upload mapping', () => {
  it('maps the optional inference_notice and normalizes blank values to null', async () => {
    const bodies = [
      {
        session_id: 3,
        image_url: '/media/a.png',
        clues: [],
        scores: [],
        action: null,
        messages: [],
        inference_notice: '  模型结果不可用，已回退到规则排序。  ',
      },
      {
        session_id: 4,
        image_url: '/media/b.png',
        clues: [],
        scores: [],
        action: null,
        messages: [],
        inference_notice: '',
      },
    ]
    let call = 0
    const fetchImpl = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify(bodies[call++]), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new GeoApi({ fetchImpl })
    const file = new File([new Uint8Array([1])], 'a.png', { type: 'image/png' })

    const degraded = await client.upload(file)
    const healthy = await client.upload(file)

    expect(degraded.inference_notice).toBe('模型结果不可用，已回退到规则排序。')
    expect(healthy.inference_notice).toBeNull()
  })
})

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

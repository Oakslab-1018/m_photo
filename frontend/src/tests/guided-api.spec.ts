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

  it('forwards a valid upload confidence route and normalizes null or invalid values', async () => {
    const confidence = {
      tier: 'high',
      action: 'direct_top1',
      model_only_margin: 0.03,
      fused_margin: 0.02,
      effective_margin: 0.02,
      prior_top_iso: 'JP',
      top1_iso: 'JP',
      top3: [
        {
          iso: 'JP',
          name: 'Japan',
          name_zh: '日本',
          score: 0.9,
          score_type: 'country_rank',
          matched_clues: [{ dimension: 'language', value: 'ja' }],
        },
      ],
      leader_changed: false,
      reason: '稳定候选',
      policy_version: 'test-policy',
    }
    const bodies = [
      {
        session_id: 5,
        image_url: '/media/confidence.png',
        clues: [],
        scores: [],
        action: { kind: 'result', payload: { iso: 'JP', confidence } },
        confidence,
        messages: [],
      },
      {
        session_id: 6,
        image_url: '/media/null.png',
        clues: [],
        scores: [],
        action: null,
        confidence: null,
        messages: [],
      },
      {
        session_id: 7,
        image_url: '/media/invalid.png',
        clues: [],
        scores: [],
        action: null,
        confidence: { tier: 'unknown', action: 'direct_top1', top3: [] },
        messages: [],
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
    const file = new File([new Uint8Array([1])], 'confidence.png', { type: 'image/png' })

    const valid = await client.upload(file)
    const nullValue = await client.upload(file)
    const invalid = await client.upload(file)

    expect(valid.confidence).toEqual(confidence)
    expect(valid.confidence?.top3[0]?.matched_clues).toEqual([
      { dimension: 'language', value: 'ja' },
    ])
    expect(nullValue.confidence).toBeNull()
    expect(invalid.confidence).toBeNull()
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

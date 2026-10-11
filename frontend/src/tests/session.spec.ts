import { describe, expect, it } from 'vitest'
import { useGeoSession } from '@/composables/useGeoSession'
import type { ChatMessage, CountryScore, UploadResponse } from '@/types'

function score(iso: string, value = 1): CountryScore {
  return { iso, name: iso, name_zh: iso, score: value, lat: 0, lng: 0 }
}

function message(content: string): ChatMessage {
  return { role: 'ai', content }
}

function upload(overrides: Partial<UploadResponse> = {}): UploadResponse {
  return {
    session_id: 1,
    image_url: '/media/a.png',
    clues: [{ dimension: 'language', value: 'ja', confidence: 0.9 }],
    scores: [score('JP', 1), score('CN', 0.5)],
    action: { kind: 'confirm', payload: { iso: 'JP' } },
    messages: [message('初始分析')],
    ...overrides,
  }
}

describe('useGeoSession', () => {
  it('applies a new upload and resets previous done/session state', () => {
    const session = useGeoSession()
    session.applyUpload(upload())
    session.applyFeedback({ done: true, iso: 'JP', scores: [], messages: [message('完成')] })
    expect(session.done.value).toBe(true)
    expect(session.confirmedIso.value).toBe('JP')

    session.applyUpload(
      upload({
        session_id: 2,
        image_url: '/media/b.png',
        scores: [score('CN')],
        action: { kind: 'ask' },
        messages: [message('新会话')],
      }),
    )

    expect(session.sessionId.value).toBe(2)
    expect(session.imageUrl.value).toBe('/media/b.png')
    expect(session.done.value).toBe(false)
    expect(session.confirmedIso.value).toBeNull()
    expect(session.exclusions.value).toEqual([])
    expect(session.scores.value.map((item) => item.iso)).toEqual(['CN'])
  })

  it('keeps current candidates when confirm returns done with empty scores', () => {
    const session = useGeoSession()
    session.applyUpload(upload())

    session.applyFeedback({
      done: true,
      iso: 'JP',
      scores: [],
      clues: [{ dimension: 'language', value: 'ja', confidence: 0.9 }],
      messages: [message('已锁定拍摄地')],
    })

    expect(session.done.value).toBe(true)
    expect(session.confirmedIso.value).toBe('JP')
    expect(session.scores.value.map((item) => item.iso)).toEqual(['JP', 'CN'])
    expect(session.topCandidate.value?.iso).toBe('JP')
    expect(session.action.value?.kind).toBe('done')
    expect(session.messages.value.at(-1)?.content).toBe('已锁定拍摄地')
    expect(session.clues.value).toHaveLength(1)
  })

  it('applies non-done feedback scores, exclusions and action', () => {
    const session = useGeoSession()
    session.applyUpload(upload())

    session.applyFeedback({
      scores: [score('CN')],
      exclusions: ['JP'],
      action: { kind: 'request_clue' },
      messages: [message('排除后重新推理')],
    })

    expect(session.scores.value.map((item) => item.iso)).toEqual(['CN'])
    expect(session.exclusions.value).toEqual(['JP'])
    expect(session.action.value?.kind).toBe('request_clue')
    expect(session.done.value).toBe(false)
    expect(session.confirmedIso.value).toBeNull()
  })

  it('applies city candidates and clears them on reset', () => {
    const session = useGeoSession()
    session.applyUpload(upload())
    expect(session.cityScores.value).toEqual([])
    expect(session.citiesStatus.value).toBeNull()

    session.applyCities({
      country_iso: 'JP',
      status: 'ok',
      city_scores: [
        {
          city_id: 'jp-tokyo', iso: 'JP', name: '东京', region: '关东',
          lat: 35.68, lng: 139.69, raw_cosine: 0.7, score: 0.7, score_type: 'city_text_cosine',
        },
      ],
    })
    expect(session.cityScores.value.map((city) => city.city_id)).toEqual(['jp-tokyo'])
    expect(session.cityCountryIso.value).toBe('JP')
    expect(session.citiesStatus.value).toBe('ok')

    session.clearCities()
    expect(session.cityScores.value).toEqual([])

    session.applyCities({
      country_iso: 'JP', status: 'unavailable', city_scores: [],
      scope: { city_limit_per_country: 5, city_catalog_revision: 'r1', cities_considered: 0 },
    })
    expect(session.citiesStatus.value).toBe('unavailable')

    session.reset()
    expect(session.cityScores.value).toEqual([])
    expect(session.cityCountryIso.value).toBeNull()
    expect(session.citiesStatus.value).toBeNull()
  })

  it('reset clears every piece of session state', () => {
    const session = useGeoSession()
    session.applyUpload(upload())

    session.reset()

    expect(session.sessionId.value).toBeNull()
    expect(session.imageUrl.value).toBeNull()
    expect(session.clues.value).toEqual([])
    expect(session.scores.value).toEqual([])
    expect(session.messages.value).toEqual([])
    expect(session.action.value).toBeNull()
    expect(session.exclusions.value).toEqual([])
    expect(session.confirmedIso.value).toBeNull()
    expect(session.done.value).toBe(false)
  })

  it('clears a stale action when a feedback response omits one', () => {
    const session = useGeoSession()
    session.applyUpload(upload())
    expect(session.action.value?.kind).toBe('confirm')

    session.applyFeedback({
      scores: [score('CN')],
      messages: [message('排除后重新推理')],
    })

    expect(session.action.value).toBeNull()
  })

  it('keeps structured answer options from a question action', () => {
    const session = useGeoSession()
    session.applyUpload(upload())

    session.applyFeedback({
      action: {
        kind: 'ask',
        payload: {
          dimensions: ['climate'],
          options: [{ dimension: 'climate', value: 'arid', label: '干旱气候' }],
        },
      },
      messages: [message('请问气候更符合哪一种？')],
    })

    expect(session.action.value?.payload?.options).toEqual([
      { dimension: 'climate', value: 'arid', label: '干旱气候' },
    ])
  })

  it('clears a stale action when point analysis omits one', () => {
    const session = useGeoSession()
    session.applyUpload(upload())

    session.applyPointAnalysis({
      scores: [score('CN')],
      messages: [message('局部区域分析完成')],
    })

    expect(session.action.value).toBeNull()
  })
})

import { describe, expect, it } from 'vitest'
import { normalizeApiError, ApiError } from '@/api/errors'

describe('normalizeApiError', () => {
  it('keeps ApiError instances unchanged', () => {
    const original = new ApiError({ message: '会话不存在', status: 404 })
    expect(normalizeApiError(original)).toBe(original)
  })

  it('prefers backend error/detail/message strings', () => {
    expect(normalizeApiError({ error: '无效的图片格式' }, 400).message).toBe('无效的图片格式')
    expect(normalizeApiError({ detail: '未登录' }, 401).message).toBe('未登录')
    expect(normalizeApiError({ message: '分析失败' }, 500).message).toBe('分析失败')
  })

  it('maps network failures to a stable Chinese message', () => {
    const normalized = normalizeApiError(new TypeError('Failed to fetch'))
    expect(normalized.message).toBe('网络连接失败，请确认后端服务已启动')
    expect(normalized.status).toBeUndefined()
  })

  it('falls back for empty or malformed payloads', () => {
    expect(normalizeApiError('', 500).message).toBe('请求失败，请稍后重试')
    expect(normalizeApiError({ unexpected: true }, 500, '上传失败').message).toBe('上传失败')
  })
})

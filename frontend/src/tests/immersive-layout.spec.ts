import { describe, expect, it } from 'vitest'
import { MIN_SAFE_MAP_EXTENT, calculateViewportInsets } from '@/composables/useImmersiveLayout'

const host = { top: 0, right: 1440, bottom: 900, left: 0, width: 1440, height: 900 }

describe('immersive map viewport insets', () => {
  it('measures command bar, both docks and a mobile sheet against the map container', () => {
    const insets = calculateViewportInsets({
      host,
      topOverlay: { top: 10, right: 1428, bottom: 62, left: 12, width: 1416, height: 52 },
      leftDock: { top: 76, right: 328, bottom: 884, left: 16, width: 312, height: 808 },
      rightDock: { top: 76, right: 1424, bottom: 884, left: 996, width: 428, height: 808 },
      bottomOverlay: { top: 610, right: 1432, bottom: 892, left: 8, width: 1424, height: 282 },
      showTop: true,
      showLeft: true,
      showRight: true,
      showBottom: true,
      clearance: 12,
    })

    expect(insets).toEqual({ top: 74, right: 456, bottom: 302, left: 340 })
  })

  it('uses actual collapsed-chip geometry and drops hidden surfaces', () => {
    const collapsed = calculateViewportInsets({
      host,
      topOverlay: { top: 10, right: 1428, bottom: 62, left: 12, width: 1416, height: 52 },
      leftDock: { top: 76, right: 116, bottom: 192, left: 16, width: 100, height: 116 },
      rightDock: { top: 76, right: 1424, bottom: 180, left: 1152, width: 272, height: 104 },
      showTop: true,
      showLeft: true,
      showRight: true,
      showBottom: false,
      clearance: 12,
    })
    const mapFocus = calculateViewportInsets({
      host,
      topOverlay: { top: 10, right: 1428, bottom: 62, left: 12, width: 1416, height: 52 },
      leftDock: { top: 76, right: 328, bottom: 884, left: 16, width: 312, height: 808 },
      rightDock: { top: 76, right: 1424, bottom: 884, left: 996, width: 428, height: 808 },
      showTop: true,
      showLeft: false,
      showRight: false,
      showBottom: false,
    })

    expect(collapsed.left).toBe(128)
    expect(collapsed.right).toBe(300)
    expect(mapFocus).toEqual({ top: 74, right: 0, bottom: 0, left: 0 })
  })

  it('clamps malformed or oversized overlays to the map rectangle', () => {
    const insets = calculateViewportInsets({
      host,
      leftDock: { top: 0, right: 2000, bottom: 100, left: -100, width: 2100, height: 100 },
      rightDock: { top: 0, right: 1600, bottom: 100, left: -50, width: 1650, height: 100 },
      showLeft: true,
      showRight: true,
      clearance: 12,
    })

    expect(insets.left).toBe(1440 - MIN_SAFE_MAP_EXTENT)
    expect(insets.right).toBe(0)
    expect(insets.top).toBe(0)
    expect(insets.bottom).toBe(0)
    expect(host.width - insets.left - insets.right).toBe(MIN_SAFE_MAP_EXTENT)
  })

  it('keeps a positive safe map window for every mobile sheet height', () => {
    const mobileHost = { top: 0, right: 390, bottom: 844, left: 0, width: 390, height: 844 }
    const topOverlay = { top: 6, right: 384, bottom: 62, left: 6, width: 378, height: 56 }
    const sideDock = { top: 66, right: 382, bottom: 836, left: 8, width: 374, height: 770 }
    const sheet = (top: number) => ({
      top,
      right: 382,
      bottom: 836,
      left: 8,
      width: 374,
      height: 836 - top,
    })
    const calculateSheet = (top: number) => calculateViewportInsets({
      host: mobileHost,
      topOverlay,
      leftDock: sideDock,
      rightDock: sideDock,
      bottomOverlay: sheet(top),
      showTop: true,
      showLeft: false,
      showRight: false,
      showBottom: true,
      clearance: 12,
    })

    const peek = calculateSheet(608)
    const half = calculateSheet(356)
    const full = calculateSheet(66)

    for (const insets of [peek, half, full]) {
      expect(insets.left).toBe(0)
      expect(insets.right).toBe(0)
      expect(mobileHost.height - insets.top - insets.bottom).toBeGreaterThanOrEqual(
        MIN_SAFE_MAP_EXTENT,
      )
    }
    expect(peek.bottom).toBe(248)
    expect(half.bottom).toBe(500)
    expect(full.top).toBe(74)
    expect(full.bottom).toBe(674)
    expect(full.bottom).toBeGreaterThan(half.bottom)
    expect(half.bottom).toBeGreaterThan(peek.bottom)
  })

  it('drops the sheet inset while the mobile evidence modal is open', () => {
    const mobileHost = { top: 0, right: 390, bottom: 844, left: 0, width: 390, height: 844 }
    const insets = calculateViewportInsets({
      host: mobileHost,
      topOverlay: { top: 6, right: 384, bottom: 62, left: 6, width: 378, height: 56 },
      leftDock: { top: 66, right: 382, bottom: 836, left: 8, width: 374, height: 770 },
      rightDock: { top: 66, right: 382, bottom: 836, left: 8, width: 374, height: 770 },
      bottomOverlay: { top: 66, right: 382, bottom: 836, left: 8, width: 374, height: 770 },
      showTop: true,
      showLeft: false,
      showRight: false,
      showBottom: false,
      clearance: 12,
    })

    expect(insets).toEqual({ top: 74, right: 0, bottom: 0, left: 0 })
  })

})
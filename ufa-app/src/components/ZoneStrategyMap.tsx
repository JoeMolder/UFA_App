import { useRef, useEffect, useState, useCallback } from 'react'
import { api, FieldZone, ZoneThrow } from '../api/client'

const CANVAS_WIDTH = 960
const CANVAS_HEIGHT = 420
const PADDING = 50
const FIELD_X_MIN = -25
const FIELD_X_MAX = 25
const FIELD_Y_MIN = 0
const FIELD_Y_MAX = 120

// Playing field (no end zones for zones): y=10..110
const ZONE_Y_MIN = 10
const ZONE_Y_MAX = 110

function ZoneStrategyMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [zones, setZones] = useState<FieldZone[]>([])
  const [loading, setLoading] = useState(false)
  const [teams, setTeams] = useState<string[]>([])
  const [selectedTeam, setSelectedTeam] = useState('__all__')
  const [hoveredZone, setHoveredZone] = useState<FieldZone | null>(null)
  const [zoomedZone, setZoomedZone] = useState<FieldZone | null>(null)
  const [totalSeqs, setTotalSeqs] = useState(0)

  const fieldLeft = PADDING
  const fieldRight = CANVAS_WIDTH - PADDING
  const fieldTop = PADDING
  const fieldBottom = CANVAS_HEIGHT - PADDING
  const fieldWidth = fieldRight - fieldLeft
  const fieldHeight = fieldBottom - fieldTop

  const fX = useCallback(
    (fieldY: number) => {
      let yMin = FIELD_Y_MIN, yMax = FIELD_Y_MAX
      if (zoomedZone) {
        const allY = [
          zoomedZone.y_range[0], zoomedZone.y_range[1],
          ...zoomedZone.throws.flatMap(t => [t.from_y, t.to_y]),
        ]
        yMin = Math.max(0, Math.min(...allY) - 8)
        yMax = Math.min(120, Math.max(...allY) + 8)
      }
      return fieldLeft + ((fieldY - yMin) / (yMax - yMin)) * fieldWidth
    },
    [fieldLeft, fieldWidth, zoomedZone]
  )
  const fY = useCallback(
    (fieldX: number) => {
      let xMin = FIELD_X_MIN, xMax = FIELD_X_MAX
      if (zoomedZone) {
        const allX = [
          zoomedZone.x_range[0], zoomedZone.x_range[1],
          ...zoomedZone.throws.flatMap(t => [t.from_x, t.to_x]),
        ]
        xMin = Math.max(-25, Math.min(...allX) - 5)
        xMax = Math.min(25, Math.max(...allX) + 5)
      }
      return fieldTop + ((fieldX - xMin) / (xMax - xMin)) * fieldHeight
    },
    [fieldTop, fieldHeight, zoomedZone]
  )

  useEffect(() => {
    api.getTeams().then(setTeams).catch(() => {})
  }, [])

  useEffect(() => {
    setLoading(true)
    const team = selectedTeam === '__all__' ? undefined : selectedTeam
    api.getZonePatterns(team, 4, 3)
      .then(res => { setZones(res.zones); setTotalSeqs(res.total) })
      .catch(() => setZones([]))
      .finally(() => setLoading(false))
  }, [selectedTeam])

  const drawArrow = useCallback((
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number, x2: number, y2: number,
    color: string, alpha: number, lw: number
  ) => {
    const headLen = 8
    const dx = x2 - x1, dy = y2 - y1
    const angle = Math.atan2(dy, dx)
    ctx.globalAlpha = alpha
    ctx.strokeStyle = color
    ctx.lineWidth = lw
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6))
    ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6))
    ctx.closePath(); ctx.fill()
    ctx.globalAlpha = 1
  }, [])

  // Draw a mini 3-throw sequence anchored to zone center, scaled to fit inside zone
  const drawZoneSequence = useCallback((
    ctx: CanvasRenderingContext2D,
    throws: ZoneThrow[],
    zoneCx: number, zoneCy: number,
    zoneW: number, zoneH: number,
    density: number
  ) => {
    if (throws.length === 0) return
    const THROW_COLORS = ['#60a5fa', '#34d399', '#fb923c']
    const scale = Math.min(Math.abs(zoneW), Math.abs(zoneH)) * 0.38  // scale sequence to fit zone

    // Normalize sequence relative to first from position
    const ox = throws[0].from_x, oy = throws[0].from_y

    // Find bounding box of sequence to scale it properly
    const xs = throws.flatMap(t => [t.from_x - ox, t.to_x - ox])
    const ys = throws.flatMap(t => [t.from_y - oy, t.to_y - oy])
    const maxSpread = Math.max(
      Math.max(...xs.map(Math.abs)),
      Math.max(...ys.map(Math.abs)),
      1
    )
    const seqScale = scale / maxSpread

    const toCanvasX = (fx: number, fy: number) =>
      zoneCx + (fy - oy) * seqScale  // y → horizontal (field is rotated)
    const toCanvasY = (fx: number, _fy: number) =>
      zoneCy + (fx - ox) * seqScale  // x → vertical

    for (let i = 0; i < throws.length; i++) {
      const t = throws[i]
      const x1 = toCanvasX(t.from_x, t.from_y)
      const y1 = toCanvasY(t.from_x, t.from_y)
      const x2 = toCanvasX(t.to_x, t.to_y)
      const y2 = toCanvasY(t.to_x, t.to_y)
      drawArrow(ctx, x1, y1, x2, y2, THROW_COLORS[i], 0.6 + density * 0.4, 1.5)
    }

    // Dot at origin
    const startX = toCanvasX(throws[0].from_x, throws[0].from_y)
    const startY = toCanvasY(throws[0].from_x, throws[0].from_y)
    ctx.beginPath(); ctx.arc(startX, startY, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'white'; ctx.globalAlpha = 0.8; ctx.fill(); ctx.globalAlpha = 1
  }, [drawArrow])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // ── Zoomed mode: single zone at full canvas scale ──
    if (zoomedZone) {
      ctx.fillStyle = '#2a5934'
      ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

      // End zone / midfield lines if they fall inside the zoomed view
      const ez1 = fX(20), ez2 = fX(100)
      if (ez1 > fieldLeft && ez1 < fieldRight) {
        ctx.fillStyle = 'rgba(255,255,255,0.06)'
        ctx.fillRect(fieldLeft, fieldTop, ez1 - fieldLeft, fieldHeight)
        ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.moveTo(ez1, fieldTop); ctx.lineTo(ez1, fieldBottom); ctx.stroke()
      }
      if (ez2 > fieldLeft && ez2 < fieldRight) {
        ctx.fillStyle = 'rgba(255,255,255,0.06)'
        ctx.fillRect(ez2, fieldTop, fieldRight - ez2, fieldHeight)
        ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.moveTo(ez2, fieldTop); ctx.lineTo(ez2, fieldBottom); ctx.stroke()
      }
      const mid = fX(60)
      if (mid > fieldLeft && mid < fieldRight) {
        ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4])
        ctx.beginPath(); ctx.moveTo(mid, fieldTop); ctx.lineTo(mid, fieldBottom); ctx.stroke()
        ctx.setLineDash([])
      }

      // Field border
      ctx.strokeStyle = 'white'; ctx.lineWidth = 2
      ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

      // Zone boundary highlight
      const zx1 = fX(zoomedZone.y_range[0]), zx2 = fX(zoomedZone.y_range[1])
      const zy1 = fY(zoomedZone.x_range[1]), zy2 = fY(zoomedZone.x_range[0])
      ctx.strokeStyle = 'rgba(255,220,50,0.6)'; ctx.lineWidth = 2
      ctx.strokeRect(zx1, zy1, zx2 - zx1, zy2 - zy1)

      // Draw throw sequence at real field scale
      if (zoomedZone.throws.length > 0) {
        const THROW_COLORS = ['#60a5fa', '#34d399', '#fb923c']
        for (let i = 0; i < zoomedZone.throws.length; i++) {
          const t = zoomedZone.throws[i]
          drawArrow(ctx, fX(t.from_y), fY(t.from_x), fX(t.to_y), fY(t.to_x), THROW_COLORS[i], 0.9, 3)
        }
        // Origin dot
        ctx.beginPath()
        ctx.arc(fX(zoomedZone.throws[0].from_y), fY(zoomedZone.throws[0].from_x), 6, 0, Math.PI * 2)
        ctx.fillStyle = 'white'; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1
      }

      // Zone info + back hint
      ctx.fillStyle = 'rgba(255,255,255,0.75)'; ctx.font = '13px monospace'; ctx.textAlign = 'left'
      ctx.fillText(`${zoomedZone.count.toLocaleString()} possession starts`, fieldLeft + 8, fieldTop + 20)
      ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '11px monospace'; ctx.textAlign = 'center'
      ctx.fillText('click anywhere to return to overview', CANVAS_WIDTH / 2, fieldBottom + 18)

      if (loading) {
        ctx.fillStyle = 'rgba(0,0,0,0.4)'
        ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)
      }
      return
    }
    // ── End zoomed mode ──

    ctx.fillStyle = '#2a5934'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // End zone shading
    const ez1 = fX(20), ez2 = fX(100)
    ctx.fillStyle = 'rgba(255,255,255,0.06)'
    ctx.fillRect(fieldLeft, fieldTop, ez1 - fieldLeft, fieldHeight)
    ctx.fillRect(ez2, fieldTop, fieldRight - ez2, fieldHeight)

    // End zone lines
    ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(ez1, fieldTop); ctx.lineTo(ez1, fieldBottom); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ez2, fieldTop); ctx.lineTo(ez2, fieldBottom); ctx.stroke()

    // Midfield dash
    const mid = fX(60)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4])
    ctx.beginPath(); ctx.moveTo(mid, fieldTop); ctx.lineTo(mid, fieldBottom); ctx.stroke()
    ctx.setLineDash([])

    // Field border
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Y-axis labels
    ctx.fillStyle = 'white'; ctx.font = '12px monospace'; ctx.textAlign = 'center'
    for (const fy of [0, 20, 40, 60, 80, 100, 120]) {
      ctx.fillText(String(fy), fX(fy), fieldBottom + 18)
    }
    ctx.textAlign = 'right'
    for (const fx of [-20, -10, 0, 10, 20]) {
      ctx.fillText(String(fx), fieldLeft - 8, fY(fx) + 4)
    }

    // Draw zones
    if (zones.length > 0) {
      for (const zone of zones) {
        const zx1 = fX(zone.y_range[0])
        const zx2 = fX(zone.y_range[1])
        const zy1 = fY(zone.x_range[1])  // note: x is vertical on canvas
        const zy2 = fY(zone.x_range[0])
        const zw = zx2 - zx1
        const zh = zy2 - zy1
        const cx = (zx1 + zx2) / 2
        const cy = (zy1 + zy2) / 2

        const isHovered = hoveredZone?.zone_id === zone.zone_id

        // Zone fill: density-based
        if (zone.count > 0) {
          ctx.fillStyle = `rgba(255,255,255,${0.04 + zone.relative_density * 0.10})`
          ctx.fillRect(zx1, zy1, zw, zh)
        }

        // Zone border
        ctx.strokeStyle = isHovered ? 'rgba(255,220,50,0.9)' : 'rgba(255,255,255,0.25)'
        ctx.lineWidth = isHovered ? 2 : 1
        ctx.strokeRect(zx1, zy1, zw, zh)

        // Mini sequence
        if (zone.throws.length > 0) {
          drawZoneSequence(ctx, zone.throws, cx, cy, zw, zh, zone.relative_density)
        }

        // Count label in corner
        if (zone.count > 0) {
          ctx.fillStyle = 'rgba(255,255,255,0.5)'
          ctx.font = '9px monospace'
          ctx.textAlign = 'left'
          ctx.fillText(zone.count.toLocaleString(), zx1 + 4, zy1 + 12)
        } else {
          ctx.fillStyle = 'rgba(255,255,255,0.2)'
          ctx.font = '9px monospace'
          ctx.textAlign = 'center'
          ctx.fillText('no data', cx, cy + 4)
        }
      }
    }

    // Loading overlay
    if (loading) {
      ctx.fillStyle = 'rgba(0,0,0,0.4)'
      ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)
      ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.font = '14px monospace'; ctx.textAlign = 'center'
      ctx.fillText('Loading...', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    }
  }, [zones, hoveredZone, zoomedZone, loading, fX, fY, drawZoneSequence, drawArrow,
    fieldLeft, fieldRight, fieldTop, fieldBottom, fieldWidth, fieldHeight])

  // Hit-test mouse position against zones
  const getZoneAtCanvas = useCallback((canvasX: number, canvasY: number): FieldZone | null => {
    for (const zone of zones) {
      const zx1 = fX(zone.y_range[0]), zx2 = fX(zone.y_range[1])
      const zy1 = fY(zone.x_range[0]), zy2 = fY(zone.x_range[1])
      if (canvasX >= Math.min(zx1, zx2) && canvasX <= Math.max(zx1, zx2) &&
          canvasY >= Math.min(zy1, zy2) && canvasY <= Math.max(zy1, zy2)) return zone
    }
    return null
  }, [zones, fX, fY])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (zoomedZone) {
      canvas.style.cursor = 'pointer'
      return
    }
    const rect = canvas.getBoundingClientRect()
    const cx = (e.clientX - rect.left) * (CANVAS_WIDTH / rect.width)
    const cy = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height)
    const z = getZoneAtCanvas(cx, cy)
    setHoveredZone(z)
    canvas.style.cursor = z && z.count > 0 ? 'pointer' : 'default'
  }, [getZoneAtCanvas, zoomedZone])

  const handleMouseLeave = useCallback(() => setHoveredZone(null), [])

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (zoomedZone) {
      setZoomedZone(null)
      return
    }
    const rect = canvas.getBoundingClientRect()
    const cx = (e.clientX - rect.left) * (CANVAS_WIDTH / rect.width)
    const cy = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height)
    const z = getZoneAtCanvas(cx, cy)
    if (z && z.count > 0) {
      setHoveredZone(null)
      setZoomedZone(z)
    }
  }, [zoomedZone, getZoneAtCanvas])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        {zoomedZone && (
          <button
            onClick={() => setZoomedZone(null)}
            style={{ padding: '6px 14px', fontSize: '13px', color: 'white', border: '1px solid #555', borderRadius: '6px', cursor: 'pointer', backgroundColor: '#2a2a3e' }}
          >
            ← Back to overview
          </button>
        )}
        <label style={{ color: '#aaa', fontSize: '13px', fontFamily: 'monospace' }}>
          Team:
          <select
            value={selectedTeam}
            onChange={e => setSelectedTeam(e.target.value)}
            style={{ marginLeft: '6px', padding: '6px 8px', fontSize: '13px', borderRadius: '6px', border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white' }}
          >
            <option value="__all__">All Teams</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        {totalSeqs > 0 && (
          <span style={{ color: '#666', fontSize: '12px', fontFamily: 'monospace' }}>
            {totalSeqs.toLocaleString()} possession starts
          </span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{ borderRadius: '8px', maxWidth: '100%', height: 'auto' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />

      {/* Hover tooltip */}
      {!zoomedZone && hoveredZone && hoveredZone.count > 0 && (
        <div style={{ fontFamily: 'monospace', fontSize: '12px', color: '#ccc', textAlign: 'center' }}>
          Zone ({hoveredZone.x_range[0].toFixed(0)} to {hoveredZone.x_range[1].toFixed(0)} wide,{' '}
          y={hoveredZone.y_range[0].toFixed(0)}–{hoveredZone.y_range[1].toFixed(0)}) —{' '}
          <strong style={{ color: 'white' }}>{hoveredZone.count.toLocaleString()}</strong> possession starts
          ({(hoveredZone.relative_density * 100).toFixed(0)}% of peak)
        </div>
      )}

      {/* Legend */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        {[['#60a5fa', 'Throw 1'], ['#34d399', 'Throw 2'], ['#fb923c', 'Throw 3']].map(([color, label]) => (
          <span key={label} style={{ color: '#aaa', fontSize: '12px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: color, display: 'inline-block' }} />
            {label}
          </span>
        ))}
        <span style={{ color: '#555', fontSize: '12px', fontFamily: 'monospace' }}>
          | Brighter zone = more possessions | Arrow size scaled to zone
        </span>
      </div>
    </div>
  )
}

export default ZoneStrategyMap

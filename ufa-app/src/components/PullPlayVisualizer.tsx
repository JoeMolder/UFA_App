import { useRef, useEffect, useState, useCallback } from 'react'
import { api, PullPlayResponse } from '../api/client'

const CANVAS_WIDTH = 900
const CANVAS_HEIGHT = 400
const PADDING = 50
const FIELD_X_MIN = -25
const FIELD_X_MAX = 25
const FIELD_Y_MIN = 0
const FIELD_Y_MAX = 120
const MARKER_RADIUS = 10

const THROW_COLORS = ['#3b82f6', '#22c55e', '#f97316'] // blue, green, orange

function PullPlayVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [pullPos, setPullPos] = useState({ x: 0, y: 20 })
  const [data, setData] = useState<PullPlayResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [teams, setTeams] = useState<string[]>([])
  const [selectedTeam, setSelectedTeam] = useState('__all__')

  const fieldLeft = PADDING
  const fieldRight = CANVAS_WIDTH - PADDING
  const fieldTop = PADDING
  const fieldBottom = CANVAS_HEIGHT - PADDING
  const fieldWidth = fieldRight - fieldLeft
  const fieldHeight = fieldBottom - fieldTop

  const fieldToCanvasX = useCallback(
    (fieldY: number) => fieldLeft + ((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * fieldWidth,
    [fieldLeft, fieldWidth]
  )
  const fieldToCanvasY = useCallback(
    (fieldX: number) => fieldTop + ((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * fieldHeight,
    [fieldTop, fieldHeight]
  )
  const canvasToFieldY = useCallback(
    (canvasX: number) => FIELD_Y_MIN + ((canvasX - fieldLeft) / fieldWidth) * (FIELD_Y_MAX - FIELD_Y_MIN),
    [fieldLeft, fieldWidth]
  )
  const canvasToFieldX = useCallback(
    (canvasY: number) => FIELD_X_MIN + ((canvasY - fieldTop) / fieldHeight) * (FIELD_X_MAX - FIELD_X_MIN),
    [fieldTop, fieldHeight]
  )

  // Fetch teams on mount
  useEffect(() => {
    api.getTeams().then(setTeams).catch(() => {})
  }, [])

  // Fetch pull play data with debounce
  useEffect(() => {
    let cancelled = false
    const timeout = setTimeout(async () => {
      try {
        setLoading(true)
        const team = selectedTeam === '__all__' ? undefined : selectedTeam
        const result = await api.getPullPlaySequence(pullPos.x, pullPos.y, team)
        if (!cancelled) setData(result)
      } catch (err) {
        console.error('Pull play fetch error:', err)
        if (!cancelled) setData(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }, 200)
    return () => { cancelled = true; clearTimeout(timeout) }
  }, [pullPos.x, pullPos.y, selectedTeam])

  // Draw arrow helper
  const drawArrow = useCallback(
    (ctx: CanvasRenderingContext2D, fromCX: number, fromCY: number, toCX: number, toCY: number, color: string, lineWidth: number) => {
      const headLen = 10
      const dx = toCX - fromCX
      const dy = toCY - fromCY
      const angle = Math.atan2(dy, dx)

      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.beginPath()
      ctx.moveTo(fromCX, fromCY)
      ctx.lineTo(toCX, toCY)
      ctx.stroke()

      // Arrowhead
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.moveTo(toCX, toCY)
      ctx.lineTo(toCX - headLen * Math.cos(angle - Math.PI / 6), toCY - headLen * Math.sin(angle - Math.PI / 6))
      ctx.lineTo(toCX - headLen * Math.cos(angle + Math.PI / 6), toCY - headLen * Math.sin(angle + Math.PI / 6))
      ctx.closePath()
      ctx.fill()
    },
    []
  )

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // Background
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // Field
    ctx.fillStyle = '#2a5934'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Endzones
    const ez1Right = fieldToCanvasX(20)
    const ez2Left = fieldToCanvasX(100)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
    ctx.fillRect(fieldLeft, fieldTop, ez1Right - fieldLeft, fieldHeight)
    ctx.fillRect(ez2Left, fieldTop, fieldRight - ez2Left, fieldHeight)

    // Endzone lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(ez1Right, fieldTop); ctx.lineTo(ez1Right, fieldBottom); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ez2Left, fieldTop); ctx.lineTo(ez2Left, fieldBottom); ctx.stroke()

    // Midfield
    const midX = fieldToCanvasX(60)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath(); ctx.moveTo(midX, fieldTop); ctx.lineTo(midX, fieldBottom); ctx.stroke()
    ctx.setLineDash([])

    // Field border
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 2
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Yard labels
    ctx.fillStyle = 'white'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    for (const fy of [0, 20, 40, 60, 80, 100, 120]) {
      ctx.fillText(String(fy), fieldToCanvasX(fy), fieldBottom + 18)
    }
    ctx.textAlign = 'right'
    for (const fx of [-20, -10, 0, 10, 20]) {
      ctx.fillText(String(fx), fieldLeft - 8, fieldToCanvasY(fx) + 4)
    }

    // Draw throw arrows
    if (data && data.throws.length > 0) {
      for (let i = 0; i < data.throws.length; i++) {
        const t = data.throws[i]
        const fromCX = fieldToCanvasX(t.from_y)
        const fromCY = fieldToCanvasY(t.from_x)
        const toCX = fieldToCanvasX(t.to_y)
        const toCY = fieldToCanvasY(t.to_x)
        const color = THROW_COLORS[i] || '#ffffff'

        drawArrow(ctx, fromCX, fromCY, toCX, toCY, color, 3)

        // Draw endpoint dot
        ctx.beginPath()
        ctx.arc(toCX, toCY, 6, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 1.5
        ctx.stroke()

        // Throw number label
        ctx.fillStyle = 'white'
        ctx.font = 'bold 10px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(i + 1), toCX, toCY)
        ctx.textBaseline = 'alphabetic'
      }

      // Draw start dot (first throw origin)
      const startCX = fieldToCanvasX(data.throws[0].from_y)
      const startCY = fieldToCanvasY(data.throws[0].from_x)
      ctx.beginPath()
      ctx.arc(startCX, startCY, 5, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }

    // Pull landing marker (draggable)
    const markerCX = fieldToCanvasX(pullPos.y)
    const markerCY = fieldToCanvasY(pullPos.x)
    ctx.beginPath()
    ctx.arc(markerCX, markerCY, MARKER_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = 'cyan'
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 3
    ctx.stroke()

    // Label for pull landing
    ctx.textAlign = 'left'
    ctx.font = '13px monospace'
    ctx.fillStyle = 'cyan'
    ctx.fillText(
      `Pull: (${pullPos.x.toFixed(0)}, ${pullPos.y.toFixed(0)})`,
      markerCX + 16, markerCY - 4
    )

    // Stats overlay
    if (data) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.beginPath()
      ctx.roundRect(fieldLeft + 8, fieldTop + 8, 200, data.throws.length > 0 ? 60 : 30, 6)
      ctx.fill()

      ctx.fillStyle = 'white'
      ctx.font = '12px monospace'
      ctx.textAlign = 'left'
      if (data.sample_size < 10) {
        ctx.fillStyle = '#f97316'
        ctx.fillText('Not enough data', fieldLeft + 16, fieldTop + 26)
      } else {
        ctx.fillText(`${data.sample_size} pulls`, fieldLeft + 16, fieldTop + 26)
        ctx.fillText(`${(data.scoring_rate * 100).toFixed(1)}% scoring rate`, fieldLeft + 16, fieldTop + 42)

        // Legend
        for (let i = 0; i < Math.min(3, data.throws.length); i++) {
          const ly = fieldTop + 56 + i * 14
          ctx.fillStyle = THROW_COLORS[i]
          ctx.fillRect(fieldLeft + 16, ly - 8, 8, 8)
          ctx.fillStyle = '#ccc'
          ctx.font = '10px monospace'
          ctx.fillText(`Throw ${i + 1}`, fieldLeft + 28, ly)
        }
      }
    }

    if (loading) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
      ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.font = '14px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Loading...', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    }
  }, [data, pullPos, loading, fieldToCanvasX, fieldToCanvasY, drawArrow, fieldLeft, fieldRight, fieldTop, fieldBottom, fieldWidth, fieldHeight])

  // Mouse helpers
  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const scaleX = CANVAS_WIDTH / rect.width
      const scaleY = CANVAS_HEIGHT / rect.height
      return { canvasX: (e.clientX - rect.left) * scaleX, canvasY: (e.clientY - rect.top) * scaleY }
    },
    []
  )

  const getFieldPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return null
      return {
        x: Math.max(FIELD_X_MIN, Math.min(FIELD_X_MAX, canvasToFieldX(coords.canvasY))),
        y: Math.max(FIELD_Y_MIN, Math.min(FIELD_Y_MAX, canvasToFieldY(coords.canvasX))),
      }
    },
    [canvasToFieldX, canvasToFieldY, getCanvasCoords]
  )

  const isOverDot = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return false
      const dx = coords.canvasX - fieldToCanvasX(pullPos.y)
      const dy = coords.canvasY - fieldToCanvasY(pullPos.x)
      return dx * dx + dy * dy <= (MARKER_RADIUS + 4) * (MARKER_RADIUS + 4)
    },
    [getCanvasCoords, fieldToCanvasX, fieldToCanvasY, pullPos]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isOverDot(e)) {
        setDragging(true)
      }
    },
    [isOverDot]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (canvas) {
        canvas.style.cursor = dragging ? 'grabbing' : isOverDot(e) ? 'grab' : 'default'
      }
      if (dragging) {
        const pos = getFieldPos(e)
        if (pos) setPullPos(pos)
      }
    },
    [dragging, getFieldPos, isOverDot]
  )

  const handleMouseUp = useCallback(() => {
    setDragging(false)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setDragging(false)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      {/* Team selector */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <label style={{ color: '#aaa', fontSize: '13px', fontFamily: 'monospace' }}>
          Team:
          <select
            value={selectedTeam}
            onChange={(e) => setSelectedTeam(e.target.value)}
            style={{
              marginLeft: '6px',
              padding: '6px 8px',
              fontSize: '13px',
              borderRadius: '6px',
              border: '1px solid #555',
              backgroundColor: '#2a2a3e',
              color: 'white',
            }}
          >
            <option value="__all__">All Teams</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{
          borderRadius: '8px',
          maxWidth: '100%',
          height: 'auto',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      <div style={{ color: '#888', fontSize: '13px' }}>
        Drag the cyan dot to change the pull landing position. Arrows show the expected first 3 throws.
      </div>
    </div>
  )
}

export default PullPlayVisualizer

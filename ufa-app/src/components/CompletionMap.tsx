import { useRef, useEffect, useState, useCallback } from 'react'
import { api, CompletionHeatmapResponse, PlayerOption } from '../api/client'

const CANVAS_WIDTH = 900
const CANVAS_HEIGHT = 400
const PADDING = 50
const FIELD_X_MIN = -25
const FIELD_X_MAX = 25
const FIELD_Y_MIN = 0
const FIELD_Y_MAX = 120

const fieldLeft = PADDING
const fieldRight = CANVAS_WIDTH - PADDING
const fieldTop = PADDING
const fieldBottom = CANVAS_HEIGHT - PADDING
const fieldWidth = fieldRight - fieldLeft
const fieldHeight = fieldBottom - fieldTop

// Green→Red: val=1 (complete) → green (hsl 120), val=0 (turnover) → red (hsl 0)
function completionColor(val: number): [number, number, number, number] {
  const hue = val * 120
  const s = 80
  const l = 50
  // hsl→rgb conversion
  const c = (1 - Math.abs(2 * l / 100 - 1)) * s / 100
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l / 100 - c / 2
  let r1 = 0, g1 = 0, b1 = 0
  if (hue < 60) { r1 = c; g1 = x }
  else if (hue < 120) { r1 = x; g1 = c }
  const alpha = Math.floor(val * 140 + 60)
  return [Math.floor((r1 + m) * 255), Math.floor((g1 + m) * 255), Math.floor((b1 + m) * 255), alpha]
}

// Probability to display color (for badge text)
function probToColor(p: number): string {
  if (p >= 0.8) return '#4ade80'   // bright green
  if (p >= 0.6) return '#facc15'   // yellow
  return '#f87171'                  // red
}

function fX(fieldY: number) {
  return fieldLeft + ((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * fieldWidth
}
function fY(fieldX: number) {
  return fieldTop + ((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * fieldHeight
}
function canvasToField(cx: number, cy: number): [number, number] {
  const fieldY = FIELD_Y_MIN + ((cx - fieldLeft) / fieldWidth) * (FIELD_Y_MAX - FIELD_Y_MIN)
  const fieldX = FIELD_X_MIN + ((cy - fieldTop) / fieldHeight) * (FIELD_X_MAX - FIELD_X_MIN)
  return [fieldX, fieldY]
}

function drawField(ctx: CanvasRenderingContext2D) {
  // Background
  ctx.fillStyle = '#1a2e1c'
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

  // Field grass
  ctx.fillStyle = '#2a5934'
  ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

  // End zone lines (y=20 and y=100)
  ctx.strokeStyle = 'rgba(255,255,255,0.5)'
  ctx.lineWidth = 1.5
  ctx.setLineDash([6, 4])
  for (const fy of [20, 100]) {
    const cx = fX(fy)
    ctx.beginPath()
    ctx.moveTo(cx, fieldTop)
    ctx.lineTo(cx, fieldBottom)
    ctx.stroke()
  }
  ctx.setLineDash([])

  // Field border
  ctx.strokeStyle = 'rgba(255,255,255,0.7)'
  ctx.lineWidth = 2
  ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

  // Yard labels
  ctx.fillStyle = 'rgba(255,255,255,0.5)'
  ctx.font = '11px monospace'
  ctx.textAlign = 'center'
  for (const fy of [0, 20, 40, 60, 80, 100, 120]) {
    const cx = fX(fy)
    ctx.fillText(String(fy), cx, fieldTop - 8)
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number,
  x2: number, y2: number,
  color: string,
  alpha: number,
  lineWidth: number
) {
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = lineWidth
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
  // Arrowhead
  const angle = Math.atan2(y2 - y1, x2 - x1)
  const len = 10
  ctx.beginPath()
  ctx.moveTo(x2, y2)
  ctx.lineTo(x2 - len * Math.cos(angle - 0.4), y2 - len * Math.sin(angle - 0.4))
  ctx.lineTo(x2 - len * Math.cos(angle + 0.4), y2 - len * Math.sin(angle + 0.4))
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

type ClickState = 'idle' | 'origin_set'

function CompletionMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const searchRef = useRef<HTMLDivElement>(null)

  const [throwers, setThrowers] = useState<PlayerOption[]>([])
  const [selectedThrower, setSelectedThrower] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)

  const [clickState, setClickState] = useState<ClickState>('idle')
  const [origin, setOrigin] = useState<[number, number] | null>(null)    // [fieldX, fieldY]
  const [target, setTarget] = useState<[number, number] | null>(null)    // [fieldX, fieldY]
  const [heatmap, setHeatmap] = useState<CompletionHeatmapResponse | null>(null)
  const [probability, setProbability] = useState<number | null>(null)

  const [loadingHeatmap, setLoadingHeatmap] = useState(false)
  const [loadingPredict, setLoadingPredict] = useState(false)
  const [hoverField, setHoverField] = useState<[number, number] | null>(null)
  const [hoverProb, setHoverProb] = useState<number | null>(null)

  // Load thrower list
  useEffect(() => {
    api.getCompletionThrowers()
      .then(list => {
        setThrowers(list)
        if (list.length > 0) {
          setSelectedThrower(list[0].id)
          setSearchQuery(list[0].name)
        }
      })
      .catch(() => {})
  }, [])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) setShowDropdown(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Draw everything on canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    drawField(ctx)

    // Draw heatmap pixels
    if (heatmap) {
      const grid = heatmap.grid  // 60 rows × 25 cols
      const nRows = grid.length      // 60 (fieldY axis → canvas X, left→right)
      const nCols = grid[0].length   // 25 (fieldX axis → canvas Y, top→bottom)
      const cellW = fieldWidth / nRows   // canvas X spans nRows (fieldY)
      const cellH = fieldHeight / nCols  // canvas Y spans nCols (fieldX)
      const imgData = ctx.createImageData(Math.ceil(fieldWidth), Math.ceil(fieldHeight))

      for (let row = 0; row < nRows; row++) {
        for (let col = 0; col < nCols; col++) {
          const val = grid[row][col]
          const [r, g, b, a] = completionColor(val)
          // row = fieldY index → canvas X; col = fieldX index → canvas Y
          const px = Math.floor(row * cellW)
          const py = Math.floor(col * cellH)
          const pw = Math.ceil(cellW)
          const ph = Math.ceil(cellH)
          for (let dy = 0; dy < ph; dy++) {
            for (let dx = 0; dx < pw; dx++) {
              const ix = px + dx
              const iy = py + dy
              if (ix < Math.ceil(fieldWidth) && iy < Math.ceil(fieldHeight)) {
                const i = (iy * Math.ceil(fieldWidth) + ix) * 4
                imgData.data[i] = r
                imgData.data[i + 1] = g
                imgData.data[i + 2] = b
                imgData.data[i + 3] = a
              }
            }
          }
        }
      }
      ctx.putImageData(imgData, fieldLeft, fieldTop)
    }

    // Re-draw field border and lines on top of heatmap
    ctx.strokeStyle = 'rgba(255,255,255,0.7)'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([6, 4])
    for (const fy of [20, 100]) {
      const cx = fX(fy)
      ctx.beginPath()
      ctx.moveTo(cx, fieldTop)
      ctx.lineTo(cx, fieldBottom)
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Origin dot
    if (origin) {
      const ox = fX(origin[1]), oy = fY(origin[0])

      // Arrow to target
      if (target) {
        const tx = fX(target[1]), ty = fY(target[0])
        drawArrow(ctx, ox, oy, tx, ty, 'rgba(255,255,255,0.8)', 0.8, 2.5)
      }

      // Cyan origin dot
      ctx.beginPath()
      ctx.arc(ox, oy, 7, 0, Math.PI * 2)
      ctx.fillStyle = '#22d3ee'
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()

      // Target dot
      if (target) {
        const tx = fX(target[1]), ty = fY(target[0])
        ctx.beginPath()
        ctx.arc(tx, ty, 5, 0, Math.PI * 2)
        ctx.fillStyle = 'white'
        ctx.fill()
      }
    }

    // Probability badge
    if (probability !== null) {
      const pct = (probability * 100).toFixed(1) + '%'
      ctx.font = 'bold 48px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      const bx = CANVAS_WIDTH / 2
      const by = CANVAS_HEIGHT / 2
      ctx.fillStyle = 'rgba(0,0,0,0.65)'
      const metrics = ctx.measureText(pct)
      const bw = metrics.width + 32
      const bh = 64
      ctx.beginPath()
      ctx.roundRect(bx - bw / 2, by - bh / 2, bw, bh, 10)
      ctx.fill()
      ctx.fillStyle = probToColor(probability)
      ctx.fillText(pct, bx, by)
    }

    // Hover % tooltip
    if (hoverProb !== null && hoverField) {
      const cx = fX(hoverField[1])
      const cy = fY(hoverField[0])
      const label = (hoverProb * 100).toFixed(1) + '%'
      ctx.font = '13px monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      const tw = ctx.measureText(label).width + 12
      const tx = Math.min(cx + 10, CANVAS_WIDTH - tw - 4)
      const ty = Math.max(cy - 24, 2)
      ctx.fillStyle = 'rgba(0,0,0,0.7)'
      ctx.fillRect(tx, ty, tw, 22)
      ctx.fillStyle = probToColor(hoverProb)
      ctx.fillText(label, tx + 6, ty + 4)
    }
  }, [heatmap, origin, target, probability, hoverProb, hoverField])

  useEffect(() => { draw() }, [draw])

  const getHoverProb = useCallback((fieldX: number, fieldY: number): number | null => {
    if (!heatmap) return null
    const grid = heatmap.grid
    const nRows = grid.length     // 60
    const nCols = grid[0].length  // 25
    const col = Math.floor(((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * nCols)
    const row = Math.floor(((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * nRows)
    if (col < 0 || col >= nCols || row < 0 || row >= nRows) return null
    return grid[row][col]
  }, [heatmap])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !heatmap) return
    const rect = canvas.getBoundingClientRect()
    const cx = (e.clientX - rect.left) * (CANVAS_WIDTH / rect.width)
    const cy = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height)
    if (cx < fieldLeft || cx > fieldRight || cy < fieldTop || cy > fieldBottom) {
      setHoverProb(null)
      setHoverField(null)
      return
    }
    const [fx, fy] = canvasToField(cx, cy)
    const p = getHoverProb(fx, fy)
    setHoverField([fx, fy])
    setHoverProb(p)
  }, [heatmap, getHoverProb])

  const handleMouseLeave = useCallback(() => {
    setHoverProb(null)
    setHoverField(null)
  }, [])

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !selectedThrower) return
    const rect = canvas.getBoundingClientRect()
    const cx = (e.clientX - rect.left) * (CANVAS_WIDTH / rect.width)
    const cy = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height)
    if (cx < fieldLeft || cx > fieldRight || cy < fieldTop || cy > fieldBottom) return

    const [fx, fy] = canvasToField(cx, cy)

    if (clickState === 'idle') {
      // First click: set origin, fetch heatmap
      setOrigin([fx, fy])
      setTarget(null)
      setProbability(null)
      setClickState('origin_set')
      setLoadingHeatmap(true)
      try {
        const data = await api.getCompletionHeatmap(selectedThrower, fx, fy)
        setHeatmap(data)
      } catch {
        setHeatmap(null)
      } finally {
        setLoadingHeatmap(false)
      }
    } else {
      // Second click: set target, fetch probability
      setTarget([fx, fy])
      setLoadingPredict(true)
      try {
        if (origin) {
          const result = await api.getCompletionPredict(selectedThrower, origin[0], origin[1], fx, fy)
          setProbability(result.probability)
        }
      } catch {
        setProbability(null)
      } finally {
        setLoadingPredict(false)
      }
    }
  }, [clickState, selectedThrower, origin])

  const handleReset = useCallback(() => {
    setClickState('idle')
    setOrigin(null)
    setTarget(null)
    setHeatmap(null)
    setProbability(null)
    setHoverProb(null)
    setHoverField(null)
  }, [])

  const handleThrowerChange = useCallback((t: PlayerOption) => {
    setSelectedThrower(t.id)
    setSearchQuery(t.name)
    setShowDropdown(false)
    handleReset()
  }, [handleReset])

  const instruction =
    clickState === 'idle'
      ? 'Click on the field to set throw origin'
      : target
      ? 'Click "Reset" to start over'
      : 'Now click a target position'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        <div ref={searchRef} style={{ position: 'relative' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setShowDropdown(true) }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search thrower..."
            style={{
              width: '200px', padding: '5px 10px', fontSize: '13px', borderRadius: '6px',
              border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white', outline: 'none',
            }}
          />
          {showDropdown && (
            <div style={{
              position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
              maxHeight: '200px', overflowY: 'auto',
              backgroundColor: '#1e1e2e', border: '1px solid #444', borderRadius: '6px', zIndex: 100,
            }}>
              {throwers
                .filter(t => t.name.toLowerCase().includes(searchQuery.toLowerCase()))
                .slice(0, 20)
                .map(t => (
                  <div
                    key={t.id}
                    onClick={() => handleThrowerChange(t)}
                    style={{
                      padding: '6px 12px', cursor: 'pointer', fontSize: '13px',
                      color: t.id === selectedThrower ? '#22d3ee' : '#ccc',
                      backgroundColor: t.id === selectedThrower ? '#2a2a3e' : 'transparent',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2a3e')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = t.id === selectedThrower ? '#2a2a3e' : 'transparent')}
                  >
                    {t.name}
                  </div>
                ))}
            </div>
          )}
        </div>

        <span style={{ color: '#22d3ee', fontSize: '13px' }}>{instruction}</span>

        {(loadingHeatmap || loadingPredict) && (
          <span style={{ color: '#888', fontSize: '13px' }}>Loading…</span>
        )}

        <button
          onClick={handleReset}
          style={{ padding: '5px 14px', fontSize: '13px', color: 'white', background: '#3a3a52', border: '1px solid #555', borderRadius: '6px', cursor: 'pointer' }}
        >
          Reset
        </button>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{ border: '1px solid #333', borderRadius: '8px', cursor: 'crosshair', maxWidth: '100%' }}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      {/* Legend */}
      <div style={{ display: 'flex', gap: '20px', alignItems: 'center', fontSize: '12px', color: '#888', fontFamily: 'monospace' }}>
        <span style={{ color: '#4ade80' }}>■ High completion %</span>
        <span style={{ color: '#f87171' }}>■ Low completion %</span>
        <span style={{ color: '#22d3ee' }}>● Origin</span>
        <span style={{ color: 'white' }}>● Target</span>
      </div>
    </div>
  )
}

export default CompletionMap

import { useRef, useEffect, useState, useCallback } from 'react'
import { api, EPVResponse } from '../api/client'

// Blue → Yellow → Red colormap (cold = low scoring chance, warm = high)
function epvColor(t: number): [number, number, number, number] {
  let r: number, g: number, b: number
  if (t < 0.5) {
    const s = t * 2
    r = Math.floor(s * 255)
    g = Math.floor(s * 180)
    b = Math.floor(255 * (1 - s * 0.8))
  } else {
    const s = (t - 0.5) * 2
    r = 255
    g = Math.floor(180 * (1 - s))
    b = 0
  }
  const a = Math.floor(t * 140 + 60)
  return [r, g, b, a]
}

const ALL_TEAMS = '__all__'

function EPVHeatmap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [data, setData] = useState<EPVResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hoverValue, setHoverValue] = useState<number | null>(null)
  const [hoverPos, setHoverPos] = useState<{ canvasX: number; canvasY: number } | null>(null)

  const [teams, setTeams] = useState<string[]>([])
  const [selectedTeam, setSelectedTeam] = useState(ALL_TEAMS)
  const [throwIdx, setThrowIdx] = useState(1)
  const [modelType, setModelType] = useState<'xgb' | 'nn'>('xgb')
  const [quarter, setQuarter] = useState<number | null>(null)

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

  // Field coordinate to canvas coordinate
  // fieldY maps to canvas X (field runs left-right on screen)
  // fieldX maps to canvas Y (field width runs top-bottom on screen)
  const fieldToCanvasX = useCallback(
    (fieldY: number) => fieldLeft + ((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * fieldWidth,
    [fieldLeft, fieldWidth]
  )
  const fieldToCanvasY = useCallback(
    (fieldX: number) => fieldTop + ((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * fieldHeight,
    [fieldTop, fieldHeight]
  )
  const canvasToFieldY = useCallback(
    (cx: number) => FIELD_Y_MIN + ((cx - fieldLeft) / fieldWidth) * (FIELD_Y_MAX - FIELD_Y_MIN),
    [fieldLeft, fieldWidth]
  )
  const canvasToFieldX = useCallback(
    (cy: number) => FIELD_X_MIN + ((cy - fieldTop) / fieldHeight) * (FIELD_X_MAX - FIELD_X_MIN),
    [fieldTop, fieldHeight]
  )

  useEffect(() => {
    api.getTeams().then(setTeams).catch(() => {})
  }, [])

  useEffect(() => {
    let cancelled = false
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const team = selectedTeam === ALL_TEAMS ? undefined : selectedTeam
        const result = await api.getEPVHeatmap(throwIdx, team, modelType, quarter ?? undefined)
        if (!cancelled) {
          setData(result)
          setLoading(false)
        }
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : 'Failed to load EPV data'
          // Check for 503 (model not trained yet)
          if (msg.includes('503') || msg.includes('not loaded')) {
            setError('EPV models not trained yet. Please run epv_model.ipynb first.')
          } else {
            setError(msg)
          }
          setLoading(false)
        }
      }
    }
    fetchData()
    return () => { cancelled = true }
  }, [throwIdx, selectedTeam, modelType, quarter])

  // Draw heatmap
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    const grid = data.grid  // shape: 60 rows (y) × 25 cols (x)
    const nRows = grid.length      // 60
    const nCols = grid[0].length   // 25

    // Find actual min/max for normalization
    let vmin = Infinity, vmax = -Infinity
    for (const row of grid) {
      for (const v of row) {
        if (v < vmin) vmin = v
        if (v > vmax) vmax = v
      }
    }
    const vRange = vmax - vmin || 1

    // Draw each grid cell as a rectangle on the canvas
    // grid[row][col]: row=0 is y=0 (own endzone), row=59 is y=120 (scoring end)
    //                 col=0 is x=-25 (left sideline), col=24 is x=25 (right sideline)
    const cellW = fieldWidth / nRows   // each y-step maps to canvas X
    const cellH = fieldHeight / nCols  // each x-step maps to canvas Y

    const imageData = ctx.createImageData(CANVAS_WIDTH, CANVAS_HEIGHT)

    for (let row = 0; row < nRows; row++) {
      for (let col = 0; col < nCols; col++) {
        const val = grid[row][col]
        const t = (val - vmin) / vRange
        const [r, g, b, a] = epvColor(t)

        // row → fieldY (0 to 120), col → fieldX (-25 to 25)
        const fieldY = FIELD_Y_MIN + (row / nRows) * (FIELD_Y_MAX - FIELD_Y_MIN)
        const fieldX = FIELD_X_MIN + (col / nCols) * (FIELD_X_MAX - FIELD_X_MIN)

        const cx0 = Math.round(fieldToCanvasX(fieldY))
        const cy0 = Math.round(fieldToCanvasY(fieldX))
        const cx1 = Math.round(fieldToCanvasX(fieldY + (FIELD_Y_MAX - FIELD_Y_MIN) / nRows))
        const cy1 = Math.round(fieldToCanvasY(fieldX + (FIELD_X_MAX - FIELD_X_MIN) / nCols))

        for (let py = Math.max(0, cy0); py < Math.min(CANVAS_HEIGHT, cy1); py++) {
          for (let px = Math.max(0, cx0); px < Math.min(CANVAS_WIDTH, cx1); px++) {
            const idx = (py * CANVAS_WIDTH + px) * 4
            imageData.data[idx] = r
            imageData.data[idx + 1] = g
            imageData.data[idx + 2] = b
            imageData.data[idx + 3] = a
          }
        }
      }
    }
    ctx.putImageData(imageData, 0, 0)

    // Draw field outline and yard lines
    ctx.strokeStyle = 'rgba(255,255,255,0.6)'
    ctx.lineWidth = 1.5
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // End zones (y=0-20 = own endzone, y=100-120 = scoring endzone)
    const endzoneLeft = fieldToCanvasX(20)
    const endzoneRight = fieldToCanvasX(100)
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'
    ctx.lineWidth = 1
    ctx.setLineDash([6, 4])
    ctx.beginPath()
    ctx.moveTo(endzoneLeft, fieldTop)
    ctx.lineTo(endzoneLeft, fieldBottom)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(endzoneRight, fieldTop)
    ctx.lineTo(endzoneRight, fieldBottom)
    ctx.stroke()
    ctx.setLineDash([])

    // Center line
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'
    const midX = fieldToCanvasX(60)
    ctx.beginPath()
    ctx.moveTo(midX, fieldTop)
    ctx.lineTo(midX, fieldBottom)
    ctx.stroke()

    // Axis labels
    ctx.fillStyle = 'rgba(200,200,200,0.8)'
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    for (const yVal of [0, 20, 40, 60, 80, 100, 120]) {
      const cx = fieldToCanvasX(yVal)
      ctx.fillText(String(yVal), cx, fieldBottom + 18)
    }
    ctx.fillText('Field Y (yd)', fieldLeft + fieldWidth / 2, fieldBottom + 34)

    ctx.textAlign = 'right'
    for (const xVal of [-20, -10, 0, 10, 20]) {
      const cy = fieldToCanvasY(xVal)
      ctx.fillText(String(xVal), fieldLeft - 6, cy + 4)
    }

    // Color legend bar
    const legendX = fieldRight + 10
    const legendY = fieldTop
    const legendH = fieldHeight
    const legendW = 14
    for (let i = 0; i < legendH; i++) {
      const t = 1 - i / legendH
      const [r, g, b] = epvColor(t)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(legendX, legendY + i, legendW, 1)
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'
    ctx.lineWidth = 1
    ctx.strokeRect(legendX, legendY, legendW, legendH)

    ctx.fillStyle = 'rgba(200,200,200,0.8)'
    ctx.font = '10px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`${(vmax * 100).toFixed(0)}%`, legendX + legendW + 3, legendY + 8)
    ctx.fillText(`${(vmin * 100).toFixed(0)}%`, legendX + legendW + 3, legendY + legendH)

    // Stats overlay
    let sumVal = 0, count = 0
    for (const row of grid) for (const v of row) { sumVal += v; count++ }
    const avgEPV = count > 0 ? sumVal / count : 0
    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    ctx.fillRect(fieldLeft + 4, fieldTop + 4, 220, 42)
    ctx.fillStyle = 'white'
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`Avg EPV: ${(avgEPV * 100).toFixed(1)}%`, fieldLeft + 10, fieldTop + 20)
    ctx.fillText(`Range: ${(vmin * 100).toFixed(1)}% – ${(vmax * 100).toFixed(1)}%`, fieldLeft + 10, fieldTop + 36)

    // Hover value
    if (hoverValue !== null && hoverPos) {
      ctx.fillStyle = 'rgba(0,0,0,0.7)'
      ctx.fillRect(hoverPos.canvasX + 10, hoverPos.canvasY - 20, 90, 20)
      ctx.fillStyle = 'white'
      ctx.font = '11px monospace'
      ctx.textAlign = 'left'
      ctx.fillText(`EPV: ${(hoverValue * 100).toFixed(1)}%`, hoverPos.canvasX + 14, hoverPos.canvasY - 5)
    }

  }, [data, hoverValue, hoverPos, fieldToCanvasX, fieldToCanvasY])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = CANVAS_WIDTH / rect.width
    const scaleY = CANVAS_HEIGHT / rect.height
    const cx = (e.clientX - rect.left) * scaleX
    const cy = (e.clientY - rect.top) * scaleY

    if (cx < fieldLeft || cx > fieldRight || cy < fieldTop || cy > fieldBottom) {
      setHoverValue(null)
      setHoverPos(null)
      return
    }

    const fieldY = canvasToFieldY(cx)
    const fieldX = canvasToFieldX(cy)

    const grid = data.grid
    const nRows = grid.length
    const nCols = grid[0].length

    const col = Math.floor(((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * nCols)
    const row = Math.floor(((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * nRows)

    const clampedRow = Math.max(0, Math.min(nRows - 1, row))
    const clampedCol = Math.max(0, Math.min(nCols - 1, col))

    setHoverValue(grid[clampedRow][clampedCol])
    setHoverPos({ canvasX: cx, canvasY: cy })
  }, [data, canvasToFieldX, canvasToFieldY, fieldLeft, fieldRight, fieldTop, fieldBottom])

  const handleMouseLeave = useCallback(() => {
    setHoverValue(null)
    setHoverPos(null)
  }, [])

  return (
    <div style={{ color: 'white' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: '16px' }}>
        {/* Throw index slider */}
        <div>
          <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>
            Throw # in possession: <strong style={{ color: 'white' }}>{throwIdx}</strong>
          </label>
          <input
            type="range"
            min={1}
            max={10}
            value={throwIdx}
            onChange={(e) => setThrowIdx(Number(e.target.value))}
            style={{ width: '200px', accentColor: '#f97316' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#888', width: '200px' }}>
            <span>1</span><span>10</span>
          </div>
        </div>

        {/* Team dropdown */}
        <div>
          <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>Team</label>
          <select
            value={selectedTeam}
            onChange={(e) => setSelectedTeam(e.target.value)}
            style={{
              padding: '6px 10px',
              fontSize: '13px',
              backgroundColor: '#2a2a3e',
              color: 'white',
              border: '1px solid #555',
              borderRadius: '4px',
              cursor: 'pointer',
              minWidth: '160px',
            }}
          >
            <option value={ALL_TEAMS}>All Teams (League Avg)</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Quarter selector */}
        <div>
          <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>Quarter</label>
          <div style={{ display: 'flex', gap: '0' }}>
            {([null, 1, 2, 3, 4] as const).map((q) => (
              <button
                key={q ?? 'all'}
                onClick={() => setQuarter(q)}
                style={{
                  padding: '6px 10px',
                  fontSize: '13px',
                  border: '1px solid #555',
                  cursor: 'pointer',
                  backgroundColor: quarter === q ? '#6366f1' : '#2a2a3e',
                  color: quarter === q ? 'white' : '#aaa',
                  borderRadius: q === null ? '4px 0 0 4px' : q === 4 ? '0 4px 4px 0' : '0',
                  transition: 'all 0.15s',
                }}
              >
                {q === null ? 'All' : `Q${q}`}
              </button>
            ))}
          </div>
        </div>

        {/* Model toggle */}
        <div>
          <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>Model</label>
          <div style={{ display: 'flex', gap: '0' }}>
            {(['xgb', 'nn'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setModelType(m)}
                style={{
                  padding: '6px 14px',
                  fontSize: '13px',
                  border: '1px solid #555',
                  cursor: 'pointer',
                  backgroundColor: modelType === m ? '#f97316' : '#2a2a3e',
                  color: modelType === m ? 'white' : '#aaa',
                  borderRadius: m === 'xgb' ? '4px 0 0 4px' : '0 4px 4px 0',
                  transition: 'all 0.15s',
                }}
              >
                {m === 'xgb' ? 'XGBoost' : 'Neural Net'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ position: 'relative' }}>
        {loading && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.6)', zIndex: 10, borderRadius: '4px',
          }}>
            <span style={{ color: '#ccc', fontSize: '14px' }}>Computing EPV grid...</span>
          </div>
        )}
        {error && (
          <div style={{
            padding: '20px', color: '#f87171', backgroundColor: '#1f1f2e',
            borderRadius: '6px', border: '1px solid #ef4444', marginBottom: '16px',
          }}>
            {error}
          </div>
        )}
        {!error && (
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{
              width: '100%',
              maxWidth: `${CANVAS_WIDTH}px`,
              background: '#1a1a2e',
              borderRadius: '6px',
              border: '1px solid #333',
              display: 'block',
              cursor: 'crosshair',
            }}
          />
        )}
      </div>

      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        Hover over the field to see EPV at that location.
        Y=0 is own endzone, Y=120 is scoring end.
        Throw 1 = first throw in a possession; higher throw index = disc already in motion.
      </div>
    </div>
  )
}

export default EPVHeatmap

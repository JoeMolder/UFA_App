import { useRef, useEffect, useState, useCallback } from 'react'
import { api, TurnoverOriginsResponse } from '../api/client'

// Green → Yellow → Red colormap for turnover rate
function rateColor(t: number): [number, number, number, number] {
  // t=0: green (low turnover rate), t=0.5: yellow, t=1: red (high rate)
  let r: number, g: number, b: number
  if (t < 0.5) {
    const s = t * 2 // 0→1 over first half
    r = Math.floor(s * 255)
    g = 200
    b = 30
  } else {
    const s = (t - 0.5) * 2 // 0→1 over second half
    r = 255
    g = Math.floor(200 * (1 - s))
    b = 30
  }
  const a = Math.floor(t * 180 + 40)
  return [r, g, b, a]
}

const ALL_PLAYERS = '__all__'

function TurnoverOriginsHeatmap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [data, setData] = useState<TurnoverOriginsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)
  const [hoverValue, setHoverValue] = useState<number | null>(null)

  const [players, setPlayers] = useState<string[]>([])
  const [selectedPlayer, setSelectedPlayer] = useState(ALL_PLAYERS)
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)

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

  // Fetch player list on mount
  useEffect(() => {
    api.getPlayers().then(setPlayers).catch(() => {})
  }, [])

  // Fetch heatmap data when player changes
  useEffect(() => {
    let cancelled = false
    const fetchData = async () => {
      try {
        setLoading(true)
        const player = selectedPlayer === ALL_PLAYERS ? undefined : selectedPlayer
        const result = await api.getTurnoverOrigins(player, 50, 60, 2.0)
        if (!cancelled) {
          setData(result)
          setLoading(false)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load data')
          setLoading(false)
        }
      }
    }
    fetchData()
    return () => { cancelled = true }
  }, [selectedPlayer])

  const handlePlayerSelect = (player: string) => {
    setSelectedPlayer(player)
    setSearchQuery(player === ALL_PLAYERS ? '' : player)
    setShowDropdown(false)
  }

  const filteredPlayers = players.filter((p) =>
    p.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getValueAtPosition = useCallback(
    (fieldX: number, fieldY: number) => {
      if (!data) return null
      const grid = data.grid
      const rows = grid.length
      const cols = grid[0].length
      const r = Math.floor(((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * rows)
      const c = Math.floor(((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * cols)
      if (r < 0 || r >= rows || c < 0 || c >= cols) return null
      return grid[r][c]
    },
    [data]
  )

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // Draw field background
    ctx.fillStyle = '#0a3d0a'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Draw heatmap
    const grid = data.grid
    const rows = grid.length
    const cols = grid[0].length

    // Scale: 0 to max rate in the grid, clamped for visibility
    let maxVal = 0
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c] > maxVal) maxVal = grid[r][c]
      }
    }

    if (maxVal > 0) {
      const imageData = ctx.createImageData(Math.ceil(fieldWidth), Math.ceil(fieldHeight))

      for (let py = 0; py < Math.ceil(fieldHeight); py++) {
        // py (vertical) = field X (-25 to 25) → grid column
        const c = Math.floor((py / fieldHeight) * cols)
        for (let px = 0; px < Math.ceil(fieldWidth); px++) {
          // px (horizontal) = field Y (0 to 120) → grid row
          const r = Math.floor((px / fieldWidth) * rows)
          if (r >= 0 && r < rows && c >= 0 && c < cols) {
            const t = Math.min(1, grid[r][c] / maxVal)
            const [cr, cg, cb, ca] = rateColor(t)
            const idx = (py * Math.ceil(fieldWidth) + px) * 4
            imageData.data[idx] = cr
            imageData.data[idx + 1] = cg
            imageData.data[idx + 2] = cb
            imageData.data[idx + 3] = ca
          }
        }
      }
      ctx.putImageData(imageData, fieldLeft, fieldTop)
    }

    // Field lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])

    const ez1 = fieldToCanvasX(20)
    const ez2 = fieldToCanvasX(100)
    ctx.beginPath()
    ctx.moveTo(ez1, fieldTop)
    ctx.lineTo(ez1, fieldBottom)
    ctx.moveTo(ez2, fieldTop)
    ctx.lineTo(ez2, fieldBottom)
    ctx.stroke()

    const mid = fieldToCanvasX(60)
    ctx.beginPath()
    ctx.moveTo(mid, fieldTop)
    ctx.lineTo(mid, fieldBottom)
    ctx.stroke()
    ctx.setLineDash([])

    // Field border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.lineWidth = 1.5
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Yard labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    for (const yd of [0, 20, 40, 60, 80, 100, 120]) {
      const cx = fieldToCanvasX(yd)
      ctx.fillText(`${yd}`, cx, fieldBottom + 16)
    }

    // Hover crosshair
    if (hoverPos) {
      const cx = fieldToCanvasX(hoverPos.y)
      const cy = fieldToCanvasY(hoverPos.x)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
      ctx.lineWidth = 0.5
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(cx, fieldTop)
      ctx.lineTo(cx, fieldBottom)
      ctx.moveTo(fieldLeft, cy)
      ctx.lineTo(fieldRight, cy)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Stats label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.font = '11px monospace'
    ctx.textAlign = 'left'
    const overallRate = data.total_throws > 0 ? ((data.total_turnovers / data.total_throws) * 100).toFixed(1) : '0'
    ctx.fillText(
      `${data.total_turnovers.toLocaleString()} turnovers / ${data.total_throws.toLocaleString()} throws (${overallRate}%)`,
      fieldLeft, fieldTop - 10
    )

    // Color legend
    const legendX = fieldRight - 160
    const legendY = fieldTop - 18
    const legendW = 120
    const legendH = 8
    for (let i = 0; i < legendW; i++) {
      const t = i / legendW
      const [lr, lg, lb] = rateColor(t)
      ctx.fillStyle = `rgb(${lr},${lg},${lb})`
      ctx.fillRect(legendX + i, legendY, 1, legendH)
    }
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.font = '9px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('0%', legendX, legendY + legendH + 10)
    ctx.textAlign = 'right'
    ctx.fillText(`${(maxVal * 100).toFixed(0)}%`, legendX + legendW, legendY + legendH + 10)

  }, [data, fieldToCanvasX, fieldToCanvasY, hoverPos, fieldLeft, fieldRight, fieldTop, fieldBottom, fieldWidth, fieldHeight])

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const scaleX = CANVAS_WIDTH / rect.width
      const scaleY = CANVAS_HEIGHT / rect.height
      const canvasX = (e.clientX - rect.left) * scaleX
      const canvasY = (e.clientY - rect.top) * scaleY

      if (canvasX >= fieldLeft && canvasX <= fieldRight && canvasY >= fieldTop && canvasY <= fieldBottom) {
        const fx = canvasToFieldX(canvasY)
        const fy = canvasToFieldY(canvasX)
        setHoverPos({ x: fx, y: fy })
        setHoverValue(getValueAtPosition(fx, fy))
      } else {
        setHoverPos(null)
        setHoverValue(null)
      }
    },
    [canvasToFieldX, canvasToFieldY, getValueAtPosition, fieldLeft, fieldRight, fieldTop, fieldBottom]
  )

  const handleMouseLeave = useCallback(() => {
    setHoverPos(null)
    setHoverValue(null)
  }, [])

  if (error) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#ff6666' }}>
        Error: {error}
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      {/* Player search */}
      <div style={{ position: 'relative', width: '300px' }}>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value)
            setShowDropdown(true)
          }}
          onFocus={() => setShowDropdown(true)}
          placeholder="Search players..."
          style={{
            width: '100%',
            padding: '8px 12px',
            fontSize: '14px',
            borderRadius: '6px',
            border: '1px solid #555',
            backgroundColor: '#2a2a3e',
            color: 'white',
            boxSizing: 'border-box',
          }}
        />
        {showDropdown && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              maxHeight: '200px',
              overflowY: 'auto',
              backgroundColor: '#2a2a3e',
              border: '1px solid #555',
              borderRadius: '0 0 6px 6px',
              zIndex: 10,
            }}
          >
            <div
              onClick={() => handlePlayerSelect(ALL_PLAYERS)}
              style={{
                padding: '6px 12px',
                cursor: 'pointer',
                color: selectedPlayer === ALL_PLAYERS ? 'cyan' : '#aaa',
                backgroundColor: selectedPlayer === ALL_PLAYERS ? '#3a3a5e' : 'transparent',
                fontStyle: 'italic',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#3a3a5e')}
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor =
                  selectedPlayer === ALL_PLAYERS ? '#3a3a5e' : 'transparent')
              }
            >
              All Players
            </div>
            {filteredPlayers.slice(0, 20).map((p) => (
              <div
                key={p}
                onClick={() => handlePlayerSelect(p)}
                style={{
                  padding: '6px 12px',
                  cursor: 'pointer',
                  color: p === selectedPlayer ? 'cyan' : 'white',
                  backgroundColor: p === selectedPlayer ? '#3a3a5e' : 'transparent',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#3a3a5e')}
                onMouseLeave={(e) =>
                  (e.currentTarget.style.backgroundColor =
                    p === selectedPlayer ? '#3a3a5e' : 'transparent')
                }
              >
                {p}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected player label */}
      <div style={{ color: 'white', fontSize: '16px', fontWeight: 'bold' }}>
        {selectedPlayer === ALL_PLAYERS ? 'All Players' : selectedPlayer} — Turnover Origins
      </div>

      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center' }}>
          <div style={{ color: '#888', fontSize: '14px', marginBottom: '12px' }}>Loading turnover origins...</div>
          <div style={{
            width: '200px',
            height: '4px',
            background: '#1a1a2e',
            borderRadius: '2px',
            margin: '0 auto',
            overflow: 'hidden',
          }}>
            <div style={{
              width: '50%',
              height: '100%',
              background: 'linear-gradient(90deg, #ff4400, #ffaa00)',
              borderRadius: '2px',
              animation: 'pulse 1.5s ease-in-out infinite',
            }} />
          </div>
        </div>
      ) : (
        <>
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{
              width: '100%',
              maxWidth: `${CANVAS_WIDTH}px`,
              cursor: 'crosshair',
              borderRadius: '8px',
              border: '1px solid #333',
            }}
          />
          {hoverPos && hoverValue !== null && (
            <div style={{
              marginTop: '8px',
              color: '#ccc',
              fontSize: '13px',
              fontFamily: 'monospace',
            }}>
              Position: ({hoverPos.x.toFixed(1)}, {hoverPos.y.toFixed(1)}) — Turnover Rate: {(hoverValue * 100).toFixed(1)}%
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default TurnoverOriginsHeatmap

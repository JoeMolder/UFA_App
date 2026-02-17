import { useRef, useEffect, useState, useCallback } from 'react'
import { api, PredictionResponse } from '../api/client'

interface ThrowHeatmapProps {
  players: string[]
}

// Hot colormap: black → red → yellow → white
function hotColor(t: number): [number, number, number, number] {
  // t is 0..1 (normalized probability)
  const r = Math.min(255, Math.floor(t * 3 * 255))
  const g = Math.min(255, Math.max(0, Math.floor((t * 3 - 1) * 255)))
  const b = Math.min(255, Math.max(0, Math.floor((t * 3 - 2) * 255)))
  const a = Math.floor(t * 200 + 20) // semi-transparent at low values
  return [r, g, b, a]
}

function ThrowHeatmap({ players }: ThrowHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedPlayer, setSelectedPlayer] = useState(players[0] || '')
  const [throwerPos, setThrowerPos] = useState({ x: 0, y: 60 })
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Canvas layout constants
  const CANVAS_WIDTH = 900
  const CANVAS_HEIGHT = 400
  const PADDING = 50
  const FIELD_X_MIN = -25
  const FIELD_X_MAX = 25
  const FIELD_Y_MIN = 0
  const FIELD_Y_MAX = 120

  // Field drawing area
  const fieldLeft = PADDING
  const fieldRight = CANVAS_WIDTH - PADDING
  const fieldTop = PADDING
  const fieldBottom = CANVAS_HEIGHT - PADDING
  const fieldWidth = fieldRight - fieldLeft
  const fieldHeight = fieldBottom - fieldTop

  // Convert field coords to canvas pixel coords
  // Field is rotated: Y (length 0-120) maps to canvas X, X (width -25..25) maps to canvas Y
  const fieldToCanvasX = useCallback(
    (fieldY: number) => fieldLeft + ((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * fieldWidth,
    [fieldLeft, fieldWidth]
  )
  const fieldToCanvasY = useCallback(
    (fieldX: number) => fieldTop + ((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * fieldHeight,
    [fieldTop, fieldHeight]
  )

  // Convert canvas pixel coords back to field coords
  const canvasToFieldY = useCallback(
    (canvasX: number) => FIELD_Y_MIN + ((canvasX - fieldLeft) / fieldWidth) * (FIELD_Y_MAX - FIELD_Y_MIN),
    [fieldLeft, fieldWidth]
  )
  const canvasToFieldX = useCallback(
    (canvasY: number) => FIELD_X_MIN + ((canvasY - fieldTop) / fieldHeight) * (FIELD_X_MAX - FIELD_X_MIN),
    [fieldTop, fieldHeight]
  )

  const GRID_SIZE = 200

  // Fetch prediction
  const fetchPrediction = useCallback(
    async (player: string, x: number, y: number) => {
      if (!player) return
      setLoading(true)
      try {
        const result = await api.predictThrows(player, x, y, GRID_SIZE)
        setPrediction(result)
      } catch (err) {
        console.error('Prediction failed:', err)
      } finally {
        setLoading(false)
      }
    },
    []
  )

  // Debounced fetch during drag
  const debouncedFetch = useCallback(
    (player: string, x: number, y: number) => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => fetchPrediction(player, x, y), 50)
    },
    [fetchPrediction]
  )

  // Initial fetch
  useEffect(() => {
    if (selectedPlayer) {
      fetchPrediction(selectedPlayer, throwerPos.x, throwerPos.y)
    }
  }, [selectedPlayer]) // eslint-disable-line react-hooks/exhaustive-deps

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // Background
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // Field background
    ctx.fillStyle = '#2a5934'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Draw heatmap if we have prediction data
    if (prediction) {
      const grid = prediction.grid
      const rows = grid.length
      const cols = grid[0].length

      // Find max for normalization
      let maxVal = 0
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (grid[r][c] > maxVal) maxVal = grid[r][c]
        }
      }

      if (maxVal > 0) {
        const cellW = fieldWidth / cols
        const cellH = fieldHeight / rows

        // Create ImageData for efficient pixel rendering
        const imageData = ctx.getImageData(fieldLeft, fieldTop, fieldWidth, fieldHeight)
        const data = imageData.data

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const t = grid[r][c] / maxVal

            // Grid maps: col→field X (width), row→field Y (length)
            // But our canvas has field Y→canvas X, field X→canvas Y
            // Grid: row is y_bins (field Y normalized), col is x_bins (field X normalized)
            // So row→canvas X direction, col→canvas Y direction
            const pixelXStart = Math.floor(r * cellH)
            const pixelYStart = Math.floor(c * cellW)
            const pixelXEnd = Math.floor((r + 1) * cellH)
            const pixelYEnd = Math.floor((c + 1) * cellW)

            // Wait - the grid is [y_bins, x_bins] where:
            // x_bins = field X normalized (0..1 → -25..25)
            // y_bins = field Y normalized (0..1 → 0..120)
            // On canvas: field Y → horizontal (canvas X), field X → vertical (canvas Y)
            // So: row (y_bins index) → canvas X, col (x_bins index) → canvas Y
            const canvasXStart = Math.floor((r / rows) * fieldWidth)
            const canvasXEnd = Math.floor(((r + 1) / rows) * fieldWidth)
            const canvasYStart = Math.floor((c / cols) * fieldHeight)
            const canvasYEnd = Math.floor(((c + 1) / cols) * fieldHeight)

            if (t > 0.01) {
              const [cr, cg, cb, ca] = hotColor(t)
              for (let py = canvasYStart; py < canvasYEnd && py < fieldHeight; py++) {
                for (let px = canvasXStart; px < canvasXEnd && px < fieldWidth; px++) {
                  const idx = (py * fieldWidth + px) * 4
                  // Blend with field green
                  const alpha = ca / 255
                  data[idx] = Math.floor(data[idx] * (1 - alpha) + cr * alpha)
                  data[idx + 1] = Math.floor(data[idx + 1] * (1 - alpha) + cg * alpha)
                  data[idx + 2] = Math.floor(data[idx + 2] * (1 - alpha) + cb * alpha)
                  data[idx + 3] = 255
                }
              }
            }
          }
        }

        ctx.putImageData(imageData, fieldLeft, fieldTop)
      }
    }

    // Field markings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1

    // Endzones (Y=0-20 and Y=100-120)
    const ez1Right = fieldToCanvasX(20)
    const ez2Left = fieldToCanvasX(100)

    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
    ctx.fillRect(fieldLeft, fieldTop, ez1Right - fieldLeft, fieldHeight)
    ctx.fillRect(ez2Left, fieldTop, fieldRight - ez2Left, fieldHeight)

    // Goal lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(ez1Right, fieldTop)
    ctx.lineTo(ez1Right, fieldBottom)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(ez2Left, fieldTop)
    ctx.lineTo(ez2Left, fieldBottom)
    ctx.stroke()

    // Midfield line
    const midX = fieldToCanvasX(60)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(midX, fieldTop)
    ctx.lineTo(midX, fieldBottom)
    ctx.stroke()
    ctx.setLineDash([])

    // Field border
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 2
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Thrower marker
    const markerCX = fieldToCanvasX(throwerPos.y)
    const markerCY = fieldToCanvasY(throwerPos.x)

    ctx.beginPath()
    ctx.arc(markerCX, markerCY, 12, 0, Math.PI * 2)
    ctx.fillStyle = 'cyan'
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 3
    ctx.stroke()

    // Labels
    ctx.fillStyle = 'white'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'

    // Y-axis labels (field Y along canvas X)
    for (const fy of [0, 20, 40, 60, 80, 100, 120]) {
      const cx = fieldToCanvasX(fy)
      ctx.fillText(String(fy), cx, fieldBottom + 18)
    }

    // X-axis labels (field X along canvas Y)
    ctx.textAlign = 'right'
    for (const fx of [-20, -10, 0, 10, 20]) {
      const cy = fieldToCanvasY(fx)
      ctx.fillText(String(fx), fieldLeft - 8, cy + 4)
    }

    // Position label
    ctx.textAlign = 'left'
    ctx.font = '14px monospace'
    ctx.fillStyle = 'cyan'
    ctx.fillText(
      `(${throwerPos.x.toFixed(1)}, ${throwerPos.y.toFixed(1)})`,
      markerCX + 18,
      markerCY - 4
    )

    // Loading indicator
    if (loading) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.font = '16px monospace'
      ctx.textAlign = 'right'
      ctx.fillText('Loading...', CANVAS_WIDTH - 10, 20)
    }
  }, [prediction, throwerPos, loading, fieldToCanvasX, fieldToCanvasY])

  // Mouse handlers for dragging
  const getFieldPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const scaleX = CANVAS_WIDTH / rect.width
      const scaleY = CANVAS_HEIGHT / rect.height
      const canvasX = (e.clientX - rect.left) * scaleX
      const canvasY = (e.clientY - rect.top) * scaleY
      const fieldX = canvasToFieldX(canvasY)
      const fieldY = canvasToFieldY(canvasX)
      // Clamp to field bounds
      return {
        x: Math.max(FIELD_X_MIN, Math.min(FIELD_X_MAX, fieldX)),
        y: Math.max(FIELD_Y_MIN, Math.min(FIELD_Y_MAX, fieldY)),
      }
    },
    [canvasToFieldX, canvasToFieldY]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getFieldPos(e)
      if (!pos) return
      setDragging(true)
      setThrowerPos(pos)
      debouncedFetch(selectedPlayer, pos.x, pos.y)
    },
    [getFieldPos, selectedPlayer, debouncedFetch]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!dragging) return
      const pos = getFieldPos(e)
      if (!pos) return
      setThrowerPos(pos)
      debouncedFetch(selectedPlayer, pos.x, pos.y)
    },
    [dragging, getFieldPos, selectedPlayer, debouncedFetch]
  )

  const handleMouseUp = useCallback(() => {
    if (dragging) {
      setDragging(false)
      if (debounceRef.current) clearTimeout(debounceRef.current)
      fetchPrediction(selectedPlayer, throwerPos.x, throwerPos.y)
    }
  }, [dragging, selectedPlayer, throwerPos, fetchPrediction])

  // Player selection
  const handlePlayerSelect = (player: string) => {
    setSelectedPlayer(player)
    setSearchQuery(player)
    setShowDropdown(false)
  }

  const filteredPlayers = players.filter((p) =>
    p.toLowerCase().includes(searchQuery.toLowerCase())
  )

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
        {showDropdown && filteredPlayers.length > 0 && (
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
        {selectedPlayer ? `${selectedPlayer} - Throw Prediction` : 'Select a player'}
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{
          cursor: dragging ? 'grabbing' : 'crosshair',
          borderRadius: '8px',
          maxWidth: '100%',
          height: 'auto',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Instructions */}
      <div style={{ color: '#888', fontSize: '13px' }}>
        Click or drag on the field to move the thrower position
      </div>
    </div>
  )
}

export default ThrowHeatmap

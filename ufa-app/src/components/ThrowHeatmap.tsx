import { useRef, useEffect, useState, useCallback } from 'react'
import { api, BatchPredictionResponse } from '../api/client'

interface ThrowHeatmapProps {
  players: string[]
}

// Hot colormap: black → red → yellow → white
function hotColor(t: number): [number, number, number, number] {
  const r = Math.min(255, Math.floor(t * 3 * 255))
  const g = Math.min(255, Math.max(0, Math.floor((t * 3 - 1) * 255)))
  const b = Math.min(255, Math.max(0, Math.floor((t * 3 - 2) * 255)))
  const a = Math.floor(t * 200 + 20)
  return [r, g, b, a]
}

function ThrowHeatmap({ players }: ThrowHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedPlayer, setSelectedPlayer] = useState(players[0] || '')
  const [throwerPos, setThrowerPos] = useState({ x: 0, y: 60 })
  const [batchData, setBatchData] = useState<BatchPredictionResponse | null>(null)
  const [currentGrid, setCurrentGrid] = useState<number[][] | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)
  const [hoverProbability, setHoverProbability] = useState<number | null>(null)
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null)
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null)
  const [drawingBox, setDrawingBox] = useState(false)
  const [selectionBox, setSelectionBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)

  // Canvas layout constants
  const CANVAS_WIDTH = 900
  const CANVAS_HEIGHT = 400
  const PADDING = 50
  const FIELD_X_MIN = -25
  const FIELD_X_MAX = 25
  const FIELD_Y_MIN = 0
  const FIELD_Y_MAX = 120
  const MARKER_RADIUS = 12

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

  // Find nearest grid cell and return its cached heatmap
  const getGridForPosition = useCallback(
    (fieldX: number, fieldY: number) => {
      if (!batchData) return null

      const { x_positions, y_positions, grids } = batchData

      let bestXi = 0
      let bestXDist = Math.abs(fieldX - x_positions[0])
      for (let i = 1; i < x_positions.length; i++) {
        const dist = Math.abs(fieldX - x_positions[i])
        if (dist < bestXDist) {
          bestXDist = dist
          bestXi = i
        }
      }

      let bestYi = 0
      let bestYDist = Math.abs(fieldY - y_positions[0])
      for (let i = 1; i < y_positions.length; i++) {
        const dist = Math.abs(fieldY - y_positions[i])
        if (dist < bestYDist) {
          bestYDist = dist
          bestYi = i
        }
      }

      const key = `${bestXi},${bestYi}`
      return grids[key] || null
    },
    [batchData]
  )

  // Get probability at a specific field position from the current heatmap grid
  const getProbabilityAtPosition = useCallback(
    (fieldX: number, fieldY: number) => {
      if (!currentGrid) return null
      const rows = currentGrid.length
      const cols = currentGrid[0].length

      // Map field position to grid indices
      const r = Math.floor(((fieldY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * rows)
      const c = Math.floor(((fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * cols)

      if (r < 0 || r >= rows || c < 0 || c >= cols) return null
      return currentGrid[r][c]
    },
    [currentGrid]
  )

  // Compute probability within a field-coordinate box
  // Grid values are probability densities from a normalizing flow,
  // so we compute fraction of total density as the probability estimate
  const getBoxProbability = useCallback(
    (x1: number, y1: number, x2: number, y2: number) => {
      if (!currentGrid) return 0
      const rows = currentGrid.length
      const cols = currentGrid[0].length

      const minX = Math.min(x1, x2)
      const maxX = Math.max(x1, x2)
      const minY = Math.min(y1, y2)
      const maxY = Math.max(y1, y2)

      // Map field coords to grid indices
      const rMin = Math.max(0, Math.floor(((minY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * rows))
      const rMax = Math.min(rows - 1, Math.floor(((maxY - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * rows))
      const cMin = Math.max(0, Math.floor(((minX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * cols))
      const cMax = Math.min(cols - 1, Math.floor(((maxX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * cols))

      // Sum density in box and total density across entire grid
      let boxSum = 0
      let totalSum = 0
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = currentGrid[r][c]
          totalSum += val
          if (r >= rMin && r <= rMax && c >= cMin && c <= cMax) {
            boxSum += val
          }
        }
      }

      if (totalSum === 0) return 0
      return boxSum / totalSum
    },
    [currentGrid]
  )

  // Fetch predictions row-by-row for real progress tracking
  useEffect(() => {
    if (!selectedPlayer) return
    let cancelled = false

    const GRID_X = 10
    const GRID_Y = 24
    const RESOLUTION = 200

    const fetchRowByRow = async () => {
      setLoading(true)
      setLoadingProgress(0)

      const xPositions = Array.from({ length: GRID_X }, (_, i) => -25 + (50 * i) / (GRID_X - 1))
      const yPositions = Array.from({ length: GRID_Y }, (_, i) => (120 * i) / (GRID_Y - 1))
      const grids: Record<string, number[][]> = {}
      const totalCalls = GRID_X * GRID_Y
      let completed = 0

      try {
        // Process one x-row at a time, all y positions in parallel per row
        for (let xi = 0; xi < GRID_X; xi++) {
          if (cancelled) return

          const rowPromises = yPositions.map(async (fieldY, yi) => {
            const result = await api.predictThrows(selectedPlayer, xPositions[xi], fieldY, RESOLUTION)
            grids[`${xi},${yi}`] = result.grid
            completed++
            if (!cancelled) {
              setLoadingProgress(Math.round((completed / totalCalls) * 100))
            }
          })

          await Promise.all(rowPromises)
        }

        if (cancelled) return

        const batchResult = {
          grids,
          x_positions: xPositions,
          y_positions: yPositions,
          extent: [-25, 25, 0, 120] as [number, number, number, number],
        }

        setBatchData(batchResult)
        const initialGrid = grids['4,11'] // roughly center
        setCurrentGrid(initialGrid || null)
      } catch (err) {
        console.error('Prediction loading failed:', err)
      } finally {
        if (!cancelled) {
          setTimeout(() => setLoading(false), 200)
        }
      }
    }

    fetchRowByRow()
    return () => { cancelled = true }
  }, [selectedPlayer])

  // Update current grid when position changes
  useEffect(() => {
    const grid = getGridForPosition(throwerPos.x, throwerPos.y)
    if (grid) setCurrentGrid(grid)
  }, [throwerPos, getGridForPosition])

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

    // Field background
    ctx.fillStyle = '#2a5934'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Draw heatmap from cached grid
    if (currentGrid) {
      const grid = currentGrid
      const rows = grid.length
      const cols = grid[0].length

      let maxVal = 0
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (grid[r][c] > maxVal) maxVal = grid[r][c]
        }
      }

      if (maxVal > 0) {
        const imageData = ctx.getImageData(fieldLeft, fieldTop, fieldWidth, fieldHeight)
        const data = imageData.data

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const t = grid[r][c] / maxVal

            const canvasXStart = Math.floor((r / rows) * fieldWidth)
            const canvasXEnd = Math.floor(((r + 1) / rows) * fieldWidth)
            const canvasYStart = Math.floor((c / cols) * fieldHeight)
            const canvasYEnd = Math.floor(((c + 1) / cols) * fieldHeight)

            if (t > 0.01) {
              const [cr, cg, cb, ca] = hotColor(t)
              for (let py = canvasYStart; py < canvasYEnd && py < fieldHeight; py++) {
                for (let px = canvasXStart; px < canvasXEnd && px < fieldWidth; px++) {
                  const idx = (py * fieldWidth + px) * 4
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

    const ez1Right = fieldToCanvasX(20)
    const ez2Left = fieldToCanvasX(100)

    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
    ctx.fillRect(fieldLeft, fieldTop, ez1Right - fieldLeft, fieldHeight)
    ctx.fillRect(ez2Left, fieldTop, fieldRight - ez2Left, fieldHeight)

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

    const midX = fieldToCanvasX(60)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(midX, fieldTop)
    ctx.lineTo(midX, fieldBottom)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.strokeStyle = 'white'
    ctx.lineWidth = 2
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    // Thrower marker
    const markerCX = fieldToCanvasX(throwerPos.y)
    const markerCY = fieldToCanvasY(throwerPos.x)

    ctx.beginPath()
    ctx.arc(markerCX, markerCY, MARKER_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = 'cyan'
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 3
    ctx.stroke()

    // Labels
    ctx.fillStyle = 'white'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    for (const fy of [0, 20, 40, 60, 80, 100, 120]) {
      const cx = fieldToCanvasX(fy)
      ctx.fillText(String(fy), cx, fieldBottom + 18)
    }
    ctx.textAlign = 'right'
    for (const fx of [-20, -10, 0, 10, 20]) {
      const cy = fieldToCanvasY(fx)
      ctx.fillText(String(fx), fieldLeft - 8, cy + 4)
    }

    // Position label next to thrower dot
    ctx.textAlign = 'left'
    ctx.font = '14px monospace'
    ctx.fillStyle = 'cyan'
    ctx.fillText(
      `(${throwerPos.x.toFixed(1)}, ${throwerPos.y.toFixed(1)})`,
      markerCX + 18,
      markerCY - 4
    )

    // Hover probability tooltip
    if (hoverPos && hoverProbability !== null && !dragging) {
      const hoverCX = fieldToCanvasX(hoverPos.y)
      const hoverCY = fieldToCanvasY(hoverPos.x)

      // Small crosshair at hover position
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(hoverCX - 6, hoverCY)
      ctx.lineTo(hoverCX + 6, hoverCY)
      ctx.moveTo(hoverCX, hoverCY - 6)
      ctx.lineTo(hoverCX, hoverCY + 6)
      ctx.stroke()

      // Probability label
      const label = `P: ${hoverProbability.toFixed(4)}`
      ctx.font = '13px monospace'
      const metrics = ctx.measureText(label)
      const padX = 6
      const padY = 4
      const boxW = metrics.width + padX * 2
      const boxH = 18 + padY * 2

      // Position tooltip above cursor, flip if near top
      let tooltipX = hoverCX - boxW / 2
      let tooltipY = hoverCY - boxH - 10
      if (tooltipY < fieldTop) tooltipY = hoverCY + 14
      if (tooltipX < fieldLeft) tooltipX = fieldLeft
      if (tooltipX + boxW > fieldRight) tooltipX = fieldRight - boxW

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
      ctx.fillRect(tooltipX, tooltipY, boxW, boxH)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
      ctx.lineWidth = 1
      ctx.strokeRect(tooltipX, tooltipY, boxW, boxH)

      ctx.fillStyle = 'white'
      ctx.textAlign = 'left'
      ctx.fillText(label, tooltipX + padX, tooltipY + padY + 14)
    }

    // Selection box (active drawing or completed)
    const activeBox = drawingBox && boxStart && boxEnd ? boxStart : null
    const activeBoxEnd = drawingBox && boxStart && boxEnd ? boxEnd : null
    const displayBox = activeBox && activeBoxEnd
      ? { x1: activeBox.x, y1: activeBox.y, x2: activeBoxEnd.x, y2: activeBoxEnd.y }
      : selectionBox

    if (displayBox) {
      const bx1 = fieldToCanvasX(Math.min(displayBox.y1, displayBox.y2))
      const bx2 = fieldToCanvasX(Math.max(displayBox.y1, displayBox.y2))
      const by1 = fieldToCanvasY(Math.min(displayBox.x1, displayBox.x2))
      const by2 = fieldToCanvasY(Math.max(displayBox.x1, displayBox.x2))

      // Fill
      ctx.fillStyle = 'rgba(255, 255, 0, 0.15)'
      ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1)

      // Border
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
      ctx.setLineDash([])

      // Probability label
      const prob = getBoxProbability(displayBox.x1, displayBox.y1, displayBox.x2, displayBox.y2)
      const pctLabel = `${(prob * 100).toFixed(1)}%`
      ctx.font = 'bold 16px monospace'
      const labelMetrics = ctx.measureText(pctLabel)
      const labelX = (bx1 + bx2) / 2
      const labelY = (by1 + by2) / 2

      // Background pill for readability
      const pillW = labelMetrics.width + 16
      const pillH = 24
      ctx.fillStyle = 'rgba(0, 0, 0, 0.75)'
      ctx.beginPath()
      ctx.roundRect(labelX - pillW / 2, labelY - pillH / 2, pillW, pillH, 6)
      ctx.fill()

      ctx.fillStyle = '#ffd700'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(pctLabel, labelX, labelY)
      ctx.textBaseline = 'alphabetic'
    }

    // Loading bar
    if (loading) {
      const barWidth = 260
      const barHeight = 18
      const barX = (CANVAS_WIDTH - barWidth) / 2
      const barY = 14

      // Dim overlay
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
      ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

      // Bar background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.beginPath()
      ctx.roundRect(barX, barY, barWidth, barHeight, 4)
      ctx.fill()

      // Bar fill
      const fillWidth = (loadingProgress / 100) * (barWidth - 4)
      if (fillWidth > 0) {
        const gradient = ctx.createLinearGradient(barX + 2, 0, barX + 2 + fillWidth, 0)
        gradient.addColorStop(0, '#0ea5e9')
        gradient.addColorStop(1, '#06b6d4')
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.roundRect(barX + 2, barY + 2, fillWidth, barHeight - 4, 3)
        ctx.fill()
      }

      // Bar border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(barX, barY, barWidth, barHeight, 4)
      ctx.stroke()

      // Label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`Loading predictions... ${Math.round(loadingProgress)}%`, CANVAS_WIDTH / 2, barY + barHeight + 16)
    }
  }, [currentGrid, throwerPos, loading, loadingProgress, hoverPos, hoverProbability, dragging, drawingBox, boxStart, boxEnd, selectionBox, getBoxProbability, fieldToCanvasX, fieldToCanvasY])

  // Get canvas coordinates from mouse event
  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const scaleX = CANVAS_WIDTH / rect.width
      const scaleY = CANVAS_HEIGHT / rect.height
      const canvasX = (e.clientX - rect.left) * scaleX
      const canvasY = (e.clientY - rect.top) * scaleY
      return { canvasX, canvasY }
    },
    []
  )

  const getFieldPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return null
      const fieldX = canvasToFieldX(coords.canvasY)
      const fieldY = canvasToFieldY(coords.canvasX)
      return {
        x: Math.max(FIELD_X_MIN, Math.min(FIELD_X_MAX, fieldX)),
        y: Math.max(FIELD_Y_MIN, Math.min(FIELD_Y_MAX, fieldY)),
      }
    },
    [canvasToFieldX, canvasToFieldY, getCanvasCoords]
  )

  // Check if mouse is over the thrower dot
  const isOverDot = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return false
      const markerCX = fieldToCanvasX(throwerPos.y)
      const markerCY = fieldToCanvasY(throwerPos.x)
      const dx = coords.canvasX - markerCX
      const dy = coords.canvasY - markerCY
      return dx * dx + dy * dy <= (MARKER_RADIUS + 4) * (MARKER_RADIUS + 4)
    },
    [getCanvasCoords, fieldToCanvasX, fieldToCanvasY, throwerPos]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isOverDot(e)) {
        setDragging(true)
        setHoverPos(null)
        setHoverProbability(null)
      } else {
        // Start drawing a selection box
        const pos = getFieldPos(e)
        if (pos) {
          setDrawingBox(true)
          setBoxStart(pos)
          setBoxEnd(pos)
          setSelectionBox(null)

          setHoverPos(null)
          setHoverProbability(null)
        }
      }
    },
    [isOverDot, getFieldPos]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getFieldPos(e)
      if (!pos) return

      if (dragging) {
        setThrowerPos(pos)
        setHoverPos(null)
        setHoverProbability(null)
      } else if (drawingBox) {
        setBoxEnd(pos)
        setHoverPos(null)
        setHoverProbability(null)
      } else {
        // Show probability tooltip when not dragging or drawing
        const prob = getProbabilityAtPosition(pos.x, pos.y)
        setHoverPos(pos)
        setHoverProbability(prob)
      }
    },
    [dragging, drawingBox, getFieldPos, getProbabilityAtPosition]
  )

  const handleMouseUp = useCallback(() => {
    if (dragging) {
      setDragging(false)
    }
    if (drawingBox && boxStart && boxEnd) {
      // Only keep the box if it's big enough (not just a click)
      const dx = Math.abs(boxEnd.x - boxStart.x)
      const dy = Math.abs(boxEnd.y - boxStart.y)
      if (dx > 1 || dy > 2) {
        const box = { x1: boxStart.x, y1: boxStart.y, x2: boxEnd.x, y2: boxEnd.y }
        setSelectionBox(box)

      } else {
        // Too small, treat as a click — clear any existing box
        setSelectionBox(null)
      }
      setDrawingBox(false)
      setBoxStart(null)
      setBoxEnd(null)
    }
  }, [dragging, drawingBox, boxStart, boxEnd, getBoxProbability])

  const handleMouseLeave = useCallback(() => {
    setDragging(false)
    if (drawingBox) {
      setDrawingBox(false)
      setBoxStart(null)
      setBoxEnd(null)
    }
    setHoverPos(null)
    setHoverProbability(null)
  }, [drawingBox])

  // Update cursor on mouse move via ref
  const handleCursorUpdate = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (dragging) {
        canvas.style.cursor = 'grabbing'
      } else if (drawingBox) {
        canvas.style.cursor = 'crosshair'
      } else if (isOverDot(e)) {
        canvas.style.cursor = 'grab'
      } else {
        canvas.style.cursor = 'crosshair'
      }
    },
    [dragging, drawingBox, isOverDot]
  )

  const handleMouseMoveWrapper = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      handleCursorUpdate(e)
      handleMouseMove(e)
    },
    [handleCursorUpdate, handleMouseMove]
  )

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
          cursor: 'crosshair',
          borderRadius: '8px',
          maxWidth: '100%',
          height: 'auto',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMoveWrapper}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />

      {/* Instructions */}
      <div style={{ color: '#888', fontSize: '13px' }}>
        Drag the cyan dot to move the thrower. Hover to see probability. Click and drag to draw a box and see total probability.
      </div>
    </div>
  )
}

export default ThrowHeatmap

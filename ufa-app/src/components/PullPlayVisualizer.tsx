import { useRef, useEffect, useState, useCallback } from 'react'
import { api, PullPlayThrow, PullPlayCluster, PullPlayHotspot } from '../api/client'

const CANVAS_WIDTH = 900
const CANVAS_HEIGHT = 400
const PADDING = 50
const FIELD_X_MIN = -25
const FIELD_X_MAX = 25
const FIELD_Y_MIN = 0
const FIELD_Y_MAX = 120
const MARKER_RADIUS = 10

const THROW_COLORS = ['#3b82f6', '#22c55e', '#f97316']
const CLUSTER_COLORS = ['#f43f5e', '#3b82f6', '#22c55e', '#f97316', '#a855f7', '#eab308']

function PullPlayVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [pullPos, setPullPos] = useState({ x: 0, y: 20 })
  const [throws, setThrows] = useState<PullPlayThrow[]>([])
  const [sampledSequences, setSampledSequences] = useState<PullPlayThrow[][]>([])
  const [activeSampleIdx, setActiveSampleIdx] = useState<number | null>(null)
  const [clusters, setClusters] = useState<PullPlayCluster[]>([])
  const [hotspots, setHotspots] = useState<PullPlayHotspot[]>([])
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [teams, setTeams] = useState<string[]>([])
  const [selectedTeam, setSelectedTeam] = useState('__all__')
  const [mode, setMode] = useState<'model' | 'average' | 'patterns'>('patterns')
  const [sampleSize, setSampleSize] = useState<number | null>(null)
  const [scoringRate, setScoringRate] = useState<number | null>(null)

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

  useEffect(() => {
    api.getTeams().then(setTeams).catch(() => {})
  }, [])

  // Fetch hotspots when team changes
  useEffect(() => {
    const team = selectedTeam === '__all__' ? undefined : selectedTeam
    api.getPullPlayHotspots(team).then(setHotspots).catch(() => setHotspots([]))
  }, [selectedTeam])

  // Fetch sequence on position/team/mode change
  useEffect(() => {
    let cancelled = false
    const timeout = setTimeout(async () => {
      try {
        setLoading(true)
        setSampledSequences([])
        setActiveSampleIdx(null)
        const team = selectedTeam === '__all__' ? undefined : selectedTeam

        if (mode === 'patterns') {
          const result = await api.getPullPlayClusters(pullPos.x, pullPos.y, team)
          if (!cancelled) {
            setClusters(result.clusters)
            setThrows([])
            setSampleSize(null)
            setScoringRate(null)
          }
        } else {
          const result = await api.getPullPlaySequence(pullPos.x, pullPos.y, team, mode)
          if (!cancelled) {
            setThrows(result.throws)
            setClusters([])
            setSampleSize(result.sample_size)
            setScoringRate(result.scoring_rate)
          }
        }
      } catch {
        if (!cancelled) { setThrows([]); setClusters([]) }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }, 200)
    return () => { cancelled = true; clearTimeout(timeout) }
  }, [pullPos.x, pullPos.y, selectedTeam, mode])

  const handleSample = useCallback(async () => {
    try {
      setLoading(true)
      const team = selectedTeam === '__all__' ? undefined : selectedTeam
      const result = await api.samplePullPlays(pullPos.x, pullPos.y, team, 8)
      setSampledSequences(result.sequences)
      setActiveSampleIdx(0)
      setThrows(result.sequences[0])
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [pullPos.x, pullPos.y, selectedTeam])

  const drawArrow = useCallback(
    (ctx: CanvasRenderingContext2D, fromCX: number, fromCY: number, toCX: number, toCY: number, color: string, alpha: number, lineWidth: number) => {
      const headLen = 10
      const dx = toCX - fromCX
      const dy = toCY - fromCY
      const angle = Math.atan2(dy, dx)
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.beginPath()
      ctx.moveTo(fromCX, fromCY)
      ctx.lineTo(toCX, toCY)
      ctx.stroke()
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.moveTo(toCX, toCY)
      ctx.lineTo(toCX - headLen * Math.cos(angle - Math.PI / 6), toCY - headLen * Math.sin(angle - Math.PI / 6))
      ctx.lineTo(toCX - headLen * Math.cos(angle + Math.PI / 6), toCY - headLen * Math.sin(angle + Math.PI / 6))
      ctx.closePath()
      ctx.fill()
      ctx.globalAlpha = 1
    },
    []
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    ctx.fillStyle = '#2a5934'
    ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

    const ez1Right = fieldToCanvasX(20)
    const ez2Left = fieldToCanvasX(100)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
    ctx.fillRect(fieldLeft, fieldTop, ez1Right - fieldLeft, fieldHeight)
    ctx.fillRect(ez2Left, fieldTop, fieldRight - ez2Left, fieldHeight)

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(ez1Right, fieldTop); ctx.lineTo(ez1Right, fieldBottom); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ez2Left, fieldTop); ctx.lineTo(ez2Left, fieldBottom); ctx.stroke()

    const midX = fieldToCanvasX(60)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath(); ctx.moveTo(midX, fieldTop); ctx.lineTo(midX, fieldBottom); ctx.stroke()
    ctx.setLineDash([])

    ctx.strokeStyle = 'white'
    ctx.lineWidth = 2
    ctx.strokeRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)

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

    // Draw pull landing hotspots as clickable origin markers
    if (hotspots.length > 0) {
      for (const h of hotspots) {
        const hcx = fieldToCanvasX(h.y)
        const hcy = fieldToCanvasY(h.x)
        const r = 4 + h.relative_freq * 6  // size 4–10 based on frequency
        const grad = ctx.createRadialGradient(hcx, hcy, 0, hcx, hcy, r * 2.5)
        grad.addColorStop(0, `rgba(251, 191, 36, ${0.3 + h.relative_freq * 0.4})`)
        grad.addColorStop(1, 'rgba(251, 191, 36, 0)')
        ctx.beginPath()
        ctx.arc(hcx, hcy, r * 2.5, 0, Math.PI * 2)
        ctx.fillStyle = grad
        ctx.fill()
        ctx.beginPath()
        ctx.arc(hcx, hcy, r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(251, 191, 36, ${0.5 + h.relative_freq * 0.4})`
        ctx.fill()
        ctx.strokeStyle = 'rgba(255,255,255,0.6)'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    // Draw faded background samples if sampling mode active
    if (sampledSequences.length > 1) {
      for (let si = 0; si < sampledSequences.length; si++) {
        if (si === activeSampleIdx) continue
        const seq = sampledSequences[si]
        for (let i = 0; i < seq.length; i++) {
          const t = seq[i]
          drawArrow(ctx, fieldToCanvasX(t.from_y), fieldToCanvasY(t.from_x),
            fieldToCanvasX(t.to_y), fieldToCanvasY(t.to_x), THROW_COLORS[i], 0.15, 2)
        }
      }
    }

    // Draw clusters (patterns mode): all archetypes simultaneously in different colors
    if (clusters.length > 0) {
      // First pass: draw glow halos for all throw endpoints (back-to-front so arrows go on top)
      for (let ci = 0; ci < clusters.length; ci++) {
        const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length]
        const seq = clusters[ci].throws
        const offsetX = seq.length > 0 ? pullPos.x - seq[0].from_x : 0
        const offsetY = seq.length > 0 ? pullPos.y - seq[0].from_y : 0
        for (let i = 0; i < seq.length; i++) {
          const t = seq[i]
          const cx = fieldToCanvasX(t.to_y + offsetY)
          const cy = fieldToCanvasY(t.to_x + offsetX)
          const isFinal = i === seq.length - 1
          const glowR = isFinal ? 22 : 14
          // Outer glow ring
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR)
          grad.addColorStop(0, color + '55')
          grad.addColorStop(1, color + '00')
          ctx.beginPath()
          ctx.arc(cx, cy, glowR, 0, Math.PI * 2)
          ctx.fillStyle = grad
          ctx.fill()
        }
      }

      // Second pass: draw arrows
      for (let ci = 0; ci < clusters.length; ci++) {
        const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length]
        const seq = clusters[ci].throws
        const offsetX = seq.length > 0 ? pullPos.x - seq[0].from_x : 0
        const offsetY = seq.length > 0 ? pullPos.y - seq[0].from_y : 0
        for (let i = 0; i < seq.length; i++) {
          const t = seq[i]
          drawArrow(ctx,
            fieldToCanvasX(t.from_y + offsetY), fieldToCanvasY(t.from_x + offsetX),
            fieldToCanvasX(t.to_y + offsetY), fieldToCanvasY(t.to_x + offsetX),
            color, 0.85, 2.5)
        }
      }

      // Third pass: draw target dots on top of arrows
      for (let ci = 0; ci < clusters.length; ci++) {
        const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length]
        const seq = clusters[ci].throws
        const offsetX = seq.length > 0 ? pullPos.x - seq[0].from_x : 0
        const offsetY = seq.length > 0 ? pullPos.y - seq[0].from_y : 0
        for (let i = 0; i < seq.length; i++) {
          const t = seq[i]
          const cx = fieldToCanvasX(t.to_y + offsetY)
          const cy = fieldToCanvasY(t.to_x + offsetX)
          const isFinal = i === seq.length - 1
          const r = isFinal ? 7 : 4
          ctx.beginPath()
          ctx.arc(cx, cy, r, 0, Math.PI * 2)
          ctx.fillStyle = color
          ctx.globalAlpha = isFinal ? 1 : 0.6
          ctx.fill()
          ctx.globalAlpha = 1
          ctx.strokeStyle = 'white'
          ctx.lineWidth = isFinal ? 1.5 : 1
          ctx.stroke()
          // Number label on the final endpoint
          if (isFinal) {
            ctx.fillStyle = 'white'
            ctx.font = 'bold 9px monospace'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(String(ci + 1), cx, cy)
            ctx.textBaseline = 'alphabetic'
          }
        }
      }
    }

    // Draw active throws (model / average mode)
    if (throws.length > 0) {
      for (let i = 0; i < throws.length; i++) {
        const t = throws[i]
        const fromCX = fieldToCanvasX(t.from_y)
        const fromCY = fieldToCanvasY(t.from_x)
        const toCX = fieldToCanvasX(t.to_y)
        const toCY = fieldToCanvasY(t.to_x)
        drawArrow(ctx, fromCX, fromCY, toCX, toCY, THROW_COLORS[i], 1, 3)

        ctx.beginPath()
        ctx.arc(toCX, toCY, 7, 0, Math.PI * 2)
        ctx.fillStyle = THROW_COLORS[i]
        ctx.fill()
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 1.5
        ctx.stroke()

        ctx.fillStyle = 'white'
        ctx.font = 'bold 10px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(i + 1), toCX, toCY)
        ctx.textBaseline = 'alphabetic'
      }

      const startCX = fieldToCanvasX(throws[0].from_y)
      const startCY = fieldToCanvasY(throws[0].from_x)
      ctx.beginPath()
      ctx.arc(startCX, startCY, 5, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }

    // Pull landing marker
    const markerCX = fieldToCanvasX(pullPos.y)
    const markerCY = fieldToCanvasY(pullPos.x)
    ctx.beginPath()
    ctx.arc(markerCX, markerCY, MARKER_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = 'cyan'
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 3
    ctx.stroke()

    ctx.textAlign = 'left'
    ctx.font = '13px monospace'
    ctx.fillStyle = 'cyan'
    ctx.fillText(`Pull: (${pullPos.x.toFixed(0)}, ${pullPos.y.toFixed(0)})`, markerCX + 16, markerCY - 4)

    // Stats overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.beginPath()
    ctx.roundRect(fieldLeft + 8, fieldTop + 8, 180, 50, 6)
    ctx.fill()
    ctx.fillStyle = 'white'
    ctx.font = '11px monospace'
    ctx.textAlign = 'left'
    const modeLabel = mode === 'model' ? 'CVAE Model' : mode === 'patterns' ? 'Common Patterns' : 'Data Average'
    ctx.fillStyle = mode === 'model' ? '#a78bfa' : mode === 'patterns' ? '#f43f5e' : '#60a5fa'
    ctx.fillText(modeLabel, fieldLeft + 16, fieldTop + 24)
    ctx.fillStyle = '#ccc'
    if (mode === 'patterns' && clusters.length > 0) {
      ctx.fillText(`${clusters.length} archetypes`, fieldLeft + 16, fieldTop + 40)
    } else if (mode === 'average' && sampleSize !== null) {
      ctx.fillText(`${sampleSize} sequences`, fieldLeft + 16, fieldTop + 40)
    } else if (mode === 'average' && sampleSize === null) {
      ctx.fillText('No data', fieldLeft + 16, fieldTop + 40)
    }
    if (scoringRate !== null && mode === 'average') {
      ctx.fillText(`${(scoringRate * 100).toFixed(1)}% scoring`, fieldLeft + 16, fieldTop + 54)
    }

    if (loading) {
      ctx.fillStyle = 'rgba(0,0,0,0.3)'
      ctx.fillRect(fieldLeft, fieldTop, fieldWidth, fieldHeight)
      ctx.fillStyle = 'rgba(255,255,255,0.8)'
      ctx.font = '14px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Loading...', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    }
  }, [throws, sampledSequences, activeSampleIdx, clusters, hotspots, pullPos, loading, mode, sampleSize, scoringRate,
    fieldToCanvasX, fieldToCanvasY, drawArrow, fieldLeft, fieldRight, fieldTop, fieldBottom, fieldWidth, fieldHeight])

  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const scaleX = CANVAS_WIDTH / rect.width
      const scaleY = CANVAS_HEIGHT / rect.height
      return { canvasX: (e.clientX - rect.left) * scaleX, canvasY: (e.clientY - rect.top) * scaleY }
    }, []
  )

  const getFieldPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return null
      return {
        x: Math.max(FIELD_X_MIN, Math.min(FIELD_X_MAX, canvasToFieldX(coords.canvasY))),
        y: Math.max(FIELD_Y_MIN, Math.min(FIELD_Y_MAX, canvasToFieldY(coords.canvasX))),
      }
    }, [canvasToFieldX, canvasToFieldY, getCanvasCoords]
  )

  const isOverPullDot = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return false
      const dx = coords.canvasX - fieldToCanvasX(pullPos.y)
      const dy = coords.canvasY - fieldToCanvasY(pullPos.x)
      return dx * dx + dy * dy <= (MARKER_RADIUS + 4) * (MARKER_RADIUS + 4)
    }, [getCanvasCoords, fieldToCanvasX, fieldToCanvasY, pullPos]
  )

  // Returns the hotspot under the cursor, if any
  const hotspotUnderCursor = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return null
      for (const h of hotspots) {
        const hcx = fieldToCanvasX(h.y)
        const hcy = fieldToCanvasY(h.x)
        const r = (4 + h.relative_freq * 6) * 2.5  // hit area = glow radius
        const dx = coords.canvasX - hcx
        const dy = coords.canvasY - hcy
        if (dx * dx + dy * dy <= r * r) return h
      }
      return null
    }, [getCanvasCoords, fieldToCanvasX, fieldToCanvasY, hotspots]
  )

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isOverPullDot(e)) setDragging(true)
  }, [isOverPullDot])

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragging) return  // ignore click at end of drag
    const h = hotspotUnderCursor(e)
    if (h) setPullPos({ x: h.x, y: h.y })
  }, [dragging, hotspotUnderCursor])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const overHotspot = !!hotspotUnderCursor(e)
    if (canvas) canvas.style.cursor = dragging ? 'grabbing' : isOverPullDot(e) ? 'grab' : overHotspot ? 'pointer' : 'default'
    if (dragging) {
      const pos = getFieldPos(e)
      if (pos) setPullPos(pos)
    }
  }, [dragging, getFieldPos, isOverPullDot, hotspotUnderCursor])

  const handleMouseUp = useCallback(() => setDragging(false), [])
  const handleMouseLeave = useCallback(() => setDragging(false), [])

  const buttonStyle = (active: boolean, color: string) => ({
    padding: '6px 14px',
    fontSize: '13px',
    backgroundColor: active ? color : '#2a2a3e',
    color: active ? 'white' : '#aaa',
    border: `1px solid ${active ? color : '#555'}`,
    borderRadius: '6px',
    cursor: 'pointer',
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
      {/* Controls row */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        {/* Team selector */}
        <label style={{ color: '#aaa', fontSize: '13px', fontFamily: 'monospace' }}>
          Team:
          <select
            value={selectedTeam}
            onChange={(e) => { setSelectedTeam(e.target.value); setSampledSequences([]); setActiveSampleIdx(null) }}
            style={{ marginLeft: '6px', padding: '6px 8px', fontSize: '13px', borderRadius: '6px', border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white' }}
          >
            <option value="__all__">All Teams</option>
            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>

        {/* Mode toggle */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button style={buttonStyle(mode === 'patterns', '#be123c')} onClick={() => { setMode('patterns'); setSampledSequences([]); setActiveSampleIdx(null) }}>
            Common Patterns
          </button>
          <button style={buttonStyle(mode === 'model', '#7c3aed')} onClick={() => { setMode('model'); setSampledSequences([]); setActiveSampleIdx(null) }}>
            CVAE Model
          </button>
          <button style={buttonStyle(mode === 'average', '#2563eb')} onClick={() => { setMode('average'); setSampledSequences([]); setActiveSampleIdx(null) }}>
            Data Average
          </button>
        </div>

        {/* Sample button (model mode only) */}
        {mode === 'model' && (
          <button
            onClick={handleSample}
            disabled={loading}
            style={{ padding: '6px 14px', fontSize: '13px', backgroundColor: '#0f766e', color: 'white', border: 'none', borderRadius: '6px', cursor: loading ? 'not-allowed' : 'pointer' }}
          >
            Sample Plays
          </button>
        )}
      </div>

      {/* Sample navigator */}
      {sampledSequences.length > 1 && activeSampleIdx !== null && (
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            onClick={() => { const i = (activeSampleIdx - 1 + sampledSequences.length) % sampledSequences.length; setActiveSampleIdx(i); setThrows(sampledSequences[i]) }}
            style={{ padding: '4px 10px', backgroundColor: '#2a2a3e', color: 'white', border: '1px solid #555', borderRadius: '4px', cursor: 'pointer' }}
          >
            ‹
          </button>
          <span style={{ color: '#aaa', fontSize: '13px', fontFamily: 'monospace' }}>
            Sample {activeSampleIdx + 1} / {sampledSequences.length}
          </span>
          <button
            onClick={() => { const i = (activeSampleIdx + 1) % sampledSequences.length; setActiveSampleIdx(i); setThrows(sampledSequences[i]) }}
            style={{ padding: '4px 10px', backgroundColor: '#2a2a3e', color: 'white', border: '1px solid #555', borderRadius: '4px', cursor: 'pointer' }}
          >
            ›
          </button>
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{ borderRadius: '8px', maxWidth: '100%', height: 'auto' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />

      {/* Legend */}
      {mode === 'patterns' && clusters.length > 0 ? (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {clusters.map((c, i) => (
            <span key={c.cluster_id} style={{ color: '#ddd', fontSize: '12px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length], display: 'inline-block' }} />
              Pattern {c.cluster_id} ({(c.frequency * 100).toFixed(0)}%)
            </span>
          ))}
          <span style={{ color: '#555', fontSize: '12px', fontFamily: 'monospace' }}>| Drag cyan dot to reposition pull</span>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          {THROW_COLORS.map((c, i) => (
            <span key={i} style={{ color: '#aaa', fontSize: '12px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: c, display: 'inline-block' }} />
              Throw {i + 1}
            </span>
          ))}
          <span style={{ color: '#666', fontSize: '12px', fontFamily: 'monospace' }}>
            | Drag the cyan dot to move the pull landing
          </span>
        </div>
      )}
    </div>
  )
}

export default PullPlayVisualizer

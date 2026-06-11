import { useState, useEffect, useMemo } from 'react'
import { GameEvent } from '../api/client'
import '../styles/FieldVisualization.css'

interface FieldVisualizationProps {
  events: GameEvent[]
  homeTeam: string
  awayTeam: string
}

interface Tooltip {
  text: string
  x: number
  y: number
}

const WALK_THROW_TYPES = new Set([18, 19, 20, 22, 23, 24])

function FieldVisualization({ events, homeTeam, awayTeam }: FieldVisualizationProps) {
  const [walkThrowIdx, setWalkThrowIdx] = useState<number | null>(null)
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)

  const throwIndices = useMemo(
    () => events.reduce<number[]>((acc, e, i) => { if (WALK_THROW_TYPES.has(e.event_type)) acc.push(i); return acc }, []),
    [events]
  )

  // Reset walk mode when point changes
  useEffect(() => { setWalkThrowIdx(null) }, [events])

  useEffect(() => {
    if (walkThrowIdx === null) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); setWalkThrowIdx(p => p !== null ? Math.min(p + 1, throwIndices.length - 1) : null) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setWalkThrowIdx(p => p !== null ? Math.max(p - 1, 0) : null) }
      else if (e.key === 'Escape') { setWalkThrowIdx(null) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [walkThrowIdx, throwIndices.length])
  // Field dimensions based on actual data coordinate system
  const FIELD_X_MIN = -27
  const FIELD_X_MAX = 27
  const FIELD_Y_MAX = 120 // Includes endzones
  const ENDZONE_Y = 20 // Endzone depth

  // Coordinates are now stored in the same physical frame (normalized at ingestion time).
  // Home team attacks toward Y=110, away team attacks toward Y=10.
  // No per-event flipping needed.
  const normalizeY = (y: number, _team: string): number => y
  const normalizeX = (x: number, _team: string): number => x

  // SVG dimensions (landscape orientation - field displayed sideways)
  const SVG_WIDTH = 900
  const SVG_HEIGHT = 350
  const PADDING = 40

  // Scale coordinates - ROTATED 90 DEGREES
  // Map field Y (length: 0-120) to SVG X (horizontal)
  // Map field X (width: -27 to 27) to SVG Y (vertical)
  const scaleX = (fieldY: number) => {
    // Map field Y from [0, 120] to SVG width (left to right)
    const normalized = fieldY / FIELD_Y_MAX
    return PADDING + normalized * (SVG_WIDTH - 2 * PADDING)
  }

  const scaleY = (fieldX: number) => {
    // Map field X from [-27, 27] to SVG height (top to bottom, centered)
    const normalized = (fieldX - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
    return PADDING + normalized * (SVG_HEIGHT - 2 * PADDING)
  }

  // Event type constants (based on actual data)
  const EVENT_TYPE_COMPLETION = 18
  const EVENT_TYPE_THROWAWAY = 22
  const EVENT_TYPE_DROP = 20
  const EVENT_TYPE_GOAL = 19
  const EVENT_TYPE_BLOCK = 11
  const EVENT_TYPE_FOUL_BY = 16
  const EVENT_TYPE_FOUL_ON = 17
  const EVENT_TYPE_CALLAHAN = 23
  const EVENT_TYPE_STALL = 24
  const EVENT_TYPE_INJURY = 25

  const renderThrowVector = (event: GameEvent, index: number) => {
    const throwerX = event.thrower_x
    const throwerY = event.thrower_y
    const receiverX = event.receiver_x
    const receiverY = event.receiver_y
    const turnoverX = event.turnover_x
    const turnoverY = event.turnover_y
    const team = event.team || ''

    // Walk mode: determine opacity and whether to render
    let opacity = 1
    if (walkThrowIdx !== null) {
      const throwPos = throwIndices.indexOf(index)
      if (throwPos === -1) return null
      if (throwPos > walkThrowIdx) return null
      opacity = throwPos === walkThrowIdx ? 1 : 0.2
    }

    const sw = 2
    const dr = 4

    // Determine if it's a turnover (drop or throwaway)
    const isDrop = event.event_type === EVENT_TYPE_DROP
    const isThrowaway = event.event_type === EVENT_TYPE_THROWAWAY
    const isGoal = event.event_type === EVENT_TYPE_GOAL
    const isTurnover = isDrop || isThrowaway || (turnoverX != null && turnoverY != null)
    const color = isTurnover ? '#ef4444' : isGoal ? '#22c55e' : '#000000'

    const handleClick = () => {
      const throwPos = throwIndices.indexOf(index)
      if (throwPos !== -1) setWalkThrowIdx(walkThrowIdx === throwPos ? null : throwPos)
    }

    // If it's a completion or goal
    if ((event.event_type === EVENT_TYPE_COMPLETION || event.event_type === EVENT_TYPE_GOAL) &&
        throwerX != null && throwerY != null && receiverX != null && receiverY != null) {
      // Normalize coordinates based on team direction
      const normThrowerX = normalizeX(throwerX, team)
      const normThrowerY = normalizeY(throwerY, team)
      const normReceiverX = normalizeX(receiverX, team)
      const normReceiverY = normalizeY(receiverY, team)

      // ROTATED: throwerY -> SVG X, throwerX -> SVG Y
      const x1 = scaleX(normThrowerY)
      const y1 = scaleY(normThrowerX)
      const x2 = scaleX(normReceiverY)
      const y2 = scaleY(normReceiverX)

      return (
        <g key={`throw-${index}`} opacity={opacity} onClick={handleClick} style={{ cursor: 'pointer' }}>
          <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={sw}
            markerEnd={isGoal ? "url(#arrowhead-green)" : "url(#arrowhead)"} />
          <circle cx={x1} cy={y1} r={dr} fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
          <circle cx={x2} cy={y2} r={dr} fill={color}
            onMouseEnter={(e) => event.receiver && setTooltip({ text: event.receiver, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.receiver && setTooltip({ text: event.receiver, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
        </g>
      )
    }

    // Handle drops (type 20) - use receiver coordinates
    if (isDrop && throwerX != null && throwerY != null && receiverX != null && receiverY != null) {
      const normThrowerX = normalizeX(throwerX, team)
      const normThrowerY = normalizeY(throwerY, team)
      const normReceiverX = normalizeX(receiverX, team)
      const normReceiverY = normalizeY(receiverY, team)
      const x1 = scaleX(normThrowerY); const y1 = scaleY(normThrowerX)
      const x2 = scaleX(normReceiverY); const y2 = scaleY(normReceiverX)

      return (
        <g key={`throw-${index}`} opacity={opacity} onClick={handleClick} style={{ cursor: 'pointer' }}>
          <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={sw} strokeDasharray="5,5" markerEnd="url(#arrowhead-red)" />
          <circle cx={x1} cy={y1} r={dr} fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
          <circle cx={x2} cy={y2} r={dr + 2} fill={color} stroke="#fff" strokeWidth="1"
            onMouseEnter={(e) => event.receiver && setTooltip({ text: `${event.receiver} (drop)`, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.receiver && setTooltip({ text: `${event.receiver} (drop)`, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
        </g>
      )
    }

    // Handle throwaways (type 22) - use turnover coordinates
    if (isThrowaway && throwerX != null && throwerY != null && turnoverX != null && turnoverY != null) {
      const normThrowerX = normalizeX(throwerX, team)
      const normThrowerY = normalizeY(throwerY, team)
      const normTurnoverX = normalizeX(turnoverX, team)
      const normTurnoverY = normalizeY(turnoverY, team)
      const x1 = scaleX(normThrowerY); const y1 = scaleY(normThrowerX)
      const x2 = scaleX(normTurnoverY); const y2 = scaleY(normTurnoverX)

      // Check for associated block (within 3 events, only injuries between)
      let blocker: string | undefined
      for (let i = 1; i <= 3 && index + i < events.length; i++) {
        const checkEvent = events[index + i]

        // Found a block from opposing team
        if (checkEvent.event_type === EVENT_TYPE_BLOCK &&
            checkEvent.team !== team &&
            checkEvent.defender) {
          blocker = checkEvent.defender
          break
        }

        // Stop if we hit a non-injury event (possession has changed)
        if (checkEvent.event_type !== EVENT_TYPE_INJURY) {
          break
        }
      }

      const throwawayTooltip = blocker
        ? `${event.thrower} (throwaway) - blocked by ${blocker}`
        : `${event.thrower} (throwaway)`

      return (
        <g
          key={`throw-${index}`}
          opacity={opacity}
          onClick={handleClick}
          style={{ cursor: 'pointer' }}
        >
          <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={sw} strokeDasharray="5,5" markerEnd="url(#arrowhead-red)" />
          <circle cx={x1} cy={y1} r={dr} fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
          <rect x={x2 - 5} y={y2 - 5} width="10" height="10" fill={color} stroke="#fff" strokeWidth="1"
            onMouseEnter={(e) => setTooltip({ text: throwawayTooltip, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setTooltip({ text: throwawayTooltip, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
        </g>
      )
    }

    // Handle stalls (type 24) - orange diamond at thrower position
    if (event.event_type === EVENT_TYPE_STALL && throwerX != null && throwerY != null) {
      const x1 = scaleX(normalizeY(throwerY, team))
      const y1 = scaleY(normalizeX(throwerX, team))
      const sz = 6
      return (
        <g key={`throw-${index}`} opacity={opacity} onClick={handleClick} style={{ cursor: 'pointer' }}>
          <rect x={x1 - sz} y={y1 - sz} width={sz * 2} height={sz * 2} fill="#f59e0b" stroke="#fff" strokeWidth="1"
            transform={`rotate(45, ${x1}, ${y1})`}
            onMouseEnter={(e) => setTooltip({ text: `${event.thrower} (stall)`, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setTooltip({ text: `${event.thrower} (stall)`, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
        </g>
      )
    }

    // Handle callahans (type 23) - purple arrow from thrower to catch position
    if (event.event_type === EVENT_TYPE_CALLAHAN && throwerX != null && throwerY != null) {
      const x1 = scaleX(normalizeY(throwerY, team))
      const y1 = scaleY(normalizeX(throwerX, team))
      const catchX = receiverX ?? turnoverX ?? null
      const catchY = receiverY ?? turnoverY ?? null

      if (catchX != null && catchY != null) {
        const x2 = scaleX(normalizeY(catchY, team))
        const y2 = scaleY(normalizeX(catchX, team))
        return (
          <g key={`throw-${index}`} opacity={opacity} onClick={handleClick} style={{ cursor: 'pointer' }}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#a855f7" strokeWidth={sw} strokeDasharray="5,5" markerEnd="url(#arrowhead-purple)" />
            <circle cx={x1} cy={y1} r={dr} fill="#a855f7"
              onMouseEnter={(e) => setTooltip({ text: `${event.thrower} (callahan thrown)`, x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setTooltip({ text: `${event.thrower} (callahan thrown)`, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setTooltip(null)} />
            <circle cx={x2} cy={y2} r={dr + 2} fill="#a855f7" stroke="#fff" strokeWidth="2"
              onMouseEnter={(e) => setTooltip({ text: `Callahan caught (${event.defender || 'unknown'})`, x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setTooltip({ text: `Callahan caught (${event.defender || 'unknown'})`, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setTooltip(null)} />
          </g>
        )
      }

      // Fallback: dot at throw origin
      return (
        <g key={`throw-${index}`} opacity={opacity} onClick={handleClick} style={{ cursor: 'pointer' }}>
          <circle cx={x1} cy={y1} r="8" fill="#a855f7" stroke="#fff" strokeWidth="2"
            onMouseEnter={(e) => setTooltip({ text: `${event.thrower} (callahan thrown)`, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setTooltip({ text: `${event.thrower} (callahan thrown)`, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)} />
        </g>
      )
    }

    // Handle fouls (types 16/17) - blue dotted line showing disc movement (yardage gained)
    if (event.event_type === EVENT_TYPE_FOUL_BY || event.event_type === EVENT_TYPE_FOUL_ON) {
      // Look backward for previous throw's end position (where disc was)
      let fromX: number | null = null
      let fromY: number | null = null
      let fromTeam = team
      for (let i = index - 1; i >= 0; i--) {
        const prev = events[i]
        if (prev.event_type === EVENT_TYPE_COMPLETION || prev.event_type === EVENT_TYPE_GOAL) {
          fromX = prev.receiver_x ?? null
          fromY = prev.receiver_y ?? null
          fromTeam = prev.team || team
          break
        }
        if (prev.event_type === EVENT_TYPE_DROP && prev.receiver_x != null) {
          fromX = prev.receiver_x
          fromY = prev.receiver_y ?? null
          fromTeam = prev.team || team
          break
        }
        if (prev.event_type === EVENT_TYPE_THROWAWAY && prev.turnover_x != null) {
          fromX = prev.turnover_x
          fromY = prev.turnover_y ?? null
          fromTeam = prev.team || team
          break
        }
      }

      // Look forward for next throw's start position (where disc ended up after foul)
      let toX: number | null = null
      let toY: number | null = null
      let toTeam = team
      for (let i = index + 1; i < events.length; i++) {
        const next = events[i]
        if (next.thrower_x != null && next.thrower_y != null &&
            (next.event_type === EVENT_TYPE_COMPLETION || next.event_type === EVENT_TYPE_GOAL ||
             next.event_type === EVENT_TYPE_DROP || next.event_type === EVENT_TYPE_THROWAWAY)) {
          toX = next.thrower_x
          toY = next.thrower_y
          toTeam = next.team || team
          break
        }
      }

      if (fromX != null && fromY != null && toX != null && toY != null) {
        const nx1 = normalizeX(fromX, fromTeam)
        const ny1 = normalizeY(fromY, fromTeam)
        const nx2 = normalizeX(toX, toTeam)
        const ny2 = normalizeY(toY, toTeam)
        const sx1 = scaleX(ny1)
        const sy1 = scaleY(nx1)
        const sx2 = scaleX(ny2)
        const sy2 = scaleY(nx2)

        return (
          <g
            key={`throw-${index}`}
            opacity={opacity}
            onClick={handleClick}
            style={{ cursor: 'pointer' }}
          >
            <line
              x1={sx1} y1={sy1} x2={sx2} y2={sy2}
              stroke="#3b82f6"
              strokeWidth="2"
              strokeDasharray="4,4"
              markerEnd="url(#arrowhead-blue)"
            />
            <circle cx={sx1} cy={sy1} r="3" fill="#3b82f6"
              onMouseEnter={(e) => setTooltip({ text: 'Foul (disc moved)', x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setTooltip({ text: 'Foul (disc moved)', x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setTooltip(null)}
            />
            <circle cx={sx2} cy={sy2} r="3" fill="#3b82f6"
              onMouseEnter={(e) => setTooltip({ text: 'Foul (disc moved)', x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setTooltip({ text: 'Foul (disc moved)', x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setTooltip(null)}
            />
          </g>
        )
      }
    }

    return null
  }

  return (
    <div className="field-container">
      <svg width={SVG_WIDTH} height={SVG_HEIGHT} className="field-svg">
        {/* Define arrowhead markers */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#000000" />
          </marker>
          <marker
            id="arrowhead-green"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#22c55e" />
          </marker>
          <marker
            id="arrowhead-red"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#ef4444" />
          </marker>
          <marker
            id="arrowhead-purple"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#a855f7" />
          </marker>
          <marker
            id="arrowhead-blue"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#3b82f6" />
          </marker>
        </defs>

        {/* Field background */}
        <rect
          x={PADDING}
          y={PADDING}
          width={SVG_WIDTH - 2 * PADDING}
          height={SVG_HEIGHT - 2 * PADDING}
          fill="#2a5934"
          stroke="#fff"
          strokeWidth="2"
        />

        {/* Left endzone (Field Y: 0-20) */}
        <rect
          x={PADDING}
          y={PADDING}
          width={(ENDZONE_Y / FIELD_Y_MAX) * (SVG_WIDTH - 2 * PADDING)}
          height={SVG_HEIGHT - 2 * PADDING}
          fill="rgba(255, 255, 255, 0.1)"
          stroke="#fff"
          strokeWidth="1"
        />

        {/* Right endzone (Field Y: 100-120) */}
        <rect
          x={SVG_WIDTH - PADDING - (ENDZONE_Y / FIELD_Y_MAX) * (SVG_WIDTH - 2 * PADDING)}
          y={PADDING}
          width={(ENDZONE_Y / FIELD_Y_MAX) * (SVG_WIDTH - 2 * PADDING)}
          height={SVG_HEIGHT - 2 * PADDING}
          fill="rgba(255, 255, 255, 0.1)"
          stroke="#fff"
          strokeWidth="1"
        />

        {/* Midfield line (Field Y = 60) */}
        <line
          x1={scaleX(60)}
          y1={PADDING}
          x2={scaleX(60)}
          y2={SVG_HEIGHT - PADDING}
          stroke="#fff"
          strokeWidth="1"
          strokeDasharray="5,5"
        />

        {/* Goal lines */}
        <line
          x1={scaleX(20)}
          y1={PADDING}
          x2={scaleX(20)}
          y2={SVG_HEIGHT - PADDING}
          stroke="#fff"
          strokeWidth="2"
        />
        <line
          x1={scaleX(100)}
          y1={PADDING}
          x2={scaleX(100)}
          y2={SVG_HEIGHT - PADDING}
          stroke="#fff"
          strokeWidth="2"
        />

        {/* Render all throws */}
        {events.map((event, index) => renderThrowVector(event, index))}
      </svg>

      <div className="field-legend">
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#000' }}></span>
          <span>Completion</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#ef4444' }}></span>
          <span>Drop (circle)</span>
        </div>
        <div className="legend-item">
          <span className="legend-square"></span>
          <span>Throwaway (square)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#f59e0b', transform: 'rotate(45deg)', borderRadius: '0' }}></span>
          <span>Stall (diamond)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#a855f7' }}></span>
          <span>Callahan</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#3b82f6' }}></span>
          <span>Foul (disc moved)</span>
        </div>
      </div>

      {walkThrowIdx !== null ? (() => {
        const ev = events[throwIndices[walkThrowIdx]]
        const teamName = ev?.team === homeTeam ? homeTeam : ev?.team === awayTeam ? awayTeam : ev?.team ?? '?'
        const typeLabel =
          ev?.event_type === EVENT_TYPE_GOAL ? 'Goal' :
          ev?.event_type === EVENT_TYPE_THROWAWAY ? 'Throwaway' :
          ev?.event_type === EVENT_TYPE_DROP ? 'Drop' :
          ev?.event_type === EVENT_TYPE_CALLAHAN ? 'Callahan' :
          ev?.event_type === EVENT_TYPE_STALL ? 'Stall' :
          'Completion'
        const thrower = ev?.thrower ?? null
        const receiver =
          ev?.event_type === EVENT_TYPE_CALLAHAN ? (ev?.defender ?? null) :
          ev?.event_type === EVENT_TYPE_THROWAWAY ? null :
          (ev?.receiver ?? null)
        const blockerEv = ev?.event_type === EVENT_TYPE_THROWAWAY
          ? events.slice(throwIndices[walkThrowIdx] + 1, throwIndices[walkThrowIdx] + 4).find(e => e.event_type === EVENT_TYPE_BLOCK)
          : null
        return (
          <div className="selection-info">
            <span style={{ fontWeight: 600, color: '#60a5fa' }}>{teamName}</span>
            <span style={{ margin: '0 6px', color: '#666' }}>·</span>
            <span style={{ color: typeLabel === 'Goal' ? '#22c55e' : typeLabel === 'Throwaway' || typeLabel === 'Drop' ? '#ef4444' : typeLabel === 'Callahan' ? '#a855f7' : '#ccc' }}>{typeLabel}</span>
            {thrower && <><span style={{ margin: '0 6px', color: '#666' }}>·</span><span>{thrower}</span></>}
            {receiver && <><span style={{ margin: '0 4px', color: '#666' }}>→</span><span>{receiver}</span></>}
            {blockerEv?.defender && <><span style={{ margin: '0 6px', color: '#666' }}>·</span><span style={{ color: '#6366f1' }}>blk: {blockerEv.defender}</span></>}
            <span style={{ margin: '0 10px', color: '#555' }}>|</span>
            <span style={{ color: '#888', fontSize: '12px' }}>Throw {walkThrowIdx + 1}/{throwIndices.length} · ← → to step · Esc exit</span>
            <button onClick={() => setWalkThrowIdx(null)} className="clear-btn" style={{ marginLeft: 8 }}>Exit</button>
          </div>
        )
      })() : (
        <div className="selection-info" style={{ color: '#666', fontSize: '12px' }}>
          Click any throw to walk through the play
        </div>
      )}

      {tooltip && (
        <div
          className="player-tooltip"
          style={{
            position: 'fixed',
            left: `${tooltip.x + 10}px`,
            top: `${tooltip.y + 10}px`,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            whiteSpace: 'nowrap'
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  )
}

export default FieldVisualization

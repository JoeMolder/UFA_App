import { useState } from 'react'
import { GameEvent } from '../api/client'
import '../styles/FieldVisualization.css'

interface FieldVisualizationProps {
  events: GameEvent[]
  homeTeam: string
  awayTeam: string
}

interface Possession {
  team: string
  startIndex: number
  endIndex: number
  events: GameEvent[]
}

interface Tooltip {
  text: string
  x: number
  y: number
}

function FieldVisualization({ events, homeTeam, awayTeam }: FieldVisualizationProps) {
  const [selectedPossession, setSelectedPossession] = useState<number | null>(null)
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)
  // Field dimensions based on actual data coordinate system
  const FIELD_X_MIN = -27
  const FIELD_X_MAX = 27
  const FIELD_Y_MIN = 0
  const FIELD_Y_MAX = 120 // Includes endzones
  const ENDZONE_Y = 20 // Endzone depth

  // Determine which team attacks which direction
  // We'll make the home team attack left-to-right, away team attacks right-to-left
  const getAttackingDirection = (team: string): 'left-to-right' | 'right-to-left' => {
    return team === homeTeam ? 'left-to-right' : 'right-to-left'
  }

  // Flip Y coordinate for teams attacking right-to-left
  const normalizeY = (y: number, team: string): number => {
    if (getAttackingDirection(team) === 'right-to-left') {
      return FIELD_Y_MAX - y
    }
    return y
  }

  // Flip X coordinate for teams attacking right-to-left (field width is mirrored)
  const normalizeX = (x: number, team: string): number => {
    if (getAttackingDirection(team) === 'right-to-left') {
      return -x  // Mirror across center line
    }
    return x
  }

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
  const EVENT_TYPE_INJURY = 25

  // Group events into possessions
  const groupIntoPossessions = (): Possession[] => {
    const possessions: Possession[] = []
    let currentTeam: string | null = null
    let possessionStart = 0

    events.forEach((event, index) => {
      // Skip events without team info
      if (!event.team) return

      // New possession if team changed
      if (currentTeam !== null && event.team !== currentTeam) {
        possessions.push({
          team: currentTeam,
          startIndex: possessionStart,
          endIndex: index - 1,
          events: events.slice(possessionStart, index)
        })
        possessionStart = index
      }

      currentTeam = event.team

      // End possession on goal, throwaway, or drop
      if (event.event_type === EVENT_TYPE_GOAL ||
          event.event_type === EVENT_TYPE_THROWAWAY ||
          event.event_type === EVENT_TYPE_DROP) {
        possessions.push({
          team: currentTeam,
          startIndex: possessionStart,
          endIndex: index,
          events: events.slice(possessionStart, index + 1)
        })
        possessionStart = index + 1
        currentTeam = null
      }
    })

    return possessions
  }

  const possessions = groupIntoPossessions()

  // Find which possession an event belongs to
  const getPossessionIndex = (eventIndex: number): number => {
    return possessions.findIndex(p =>
      eventIndex >= p.startIndex && eventIndex <= p.endIndex
    )
  }

  const renderThrowVector = (event: GameEvent, index: number) => {
    const throwerX = event.thrower_x
    const throwerY = event.thrower_y
    const receiverX = event.receiver_x
    const receiverY = event.receiver_y
    const turnoverX = event.turnover_x
    const turnoverY = event.turnover_y
    const team = event.team || ''

    // Determine which possession this event belongs to
    const possessionIndex = getPossessionIndex(index)
    const isSelected = selectedPossession === null || selectedPossession === possessionIndex
    const opacity = isSelected ? 1 : 0.3

    // Determine if it's a turnover (drop or throwaway)
    const isDrop = event.event_type === EVENT_TYPE_DROP
    const isThrowaway = event.event_type === EVENT_TYPE_THROWAWAY
    const isGoal = event.event_type === EVENT_TYPE_GOAL
    const isTurnover = isDrop || isThrowaway || (turnoverX != null && turnoverY != null)
    const color = isTurnover ? '#ef4444' : isGoal ? '#22c55e' : '#000000' // red for turnovers, green for goals, black for completions

    // Click handler to select this possession
    const handleClick = () => {
      if (selectedPossession === possessionIndex) {
        setSelectedPossession(null) // Deselect if already selected
      } else {
        setSelectedPossession(possessionIndex)
      }
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
        <g
          key={`throw-${index}`}
          opacity={opacity}
          onClick={handleClick}
          style={{ cursor: 'pointer' }}
        >
          {/* Throw line */}
          <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={color}
            strokeWidth="2"
            markerEnd={isGoal ? "url(#arrowhead-green)" : "url(#arrowhead)"}
          />
          {/* Thrower dot */}
          <circle
            cx={x1}
            cy={y1}
            r="4"
            fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
          {/* Receiver dot */}
          <circle
            cx={x2}
            cy={y2}
            r="4"
            fill={color}
            onMouseEnter={(e) => event.receiver && setTooltip({ text: event.receiver, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.receiver && setTooltip({ text: event.receiver, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
        </g>
      )
    }

    // Handle drops (type 20) - use receiver coordinates
    if (isDrop && throwerX != null && throwerY != null && receiverX != null && receiverY != null) {
      // Normalize coordinates based on team direction
      const normThrowerX = normalizeX(throwerX, team)
      const normThrowerY = normalizeY(throwerY, team)
      const normReceiverX = normalizeX(receiverX, team)
      const normReceiverY = normalizeY(receiverY, team)

      const x1 = scaleX(normThrowerY)
      const y1 = scaleY(normThrowerX)
      const x2 = scaleX(normReceiverY)
      const y2 = scaleY(normReceiverX)

      return (
        <g
          key={`throw-${index}`}
          opacity={opacity}
          onClick={handleClick}
          style={{ cursor: 'pointer' }}
        >
          {/* Throw line */}
          <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={color}
            strokeWidth="2"
            strokeDasharray="5,5"
            markerEnd="url(#arrowhead-red)"
          />
          {/* Thrower dot */}
          <circle
            cx={x1}
            cy={y1}
            r="4"
            fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
          {/* Drop location - red circle at receiver position */}
          <circle
            cx={x2}
            cy={y2}
            r="6"
            fill={color}
            stroke="#fff"
            strokeWidth="1"
            onMouseEnter={(e) => event.receiver && setTooltip({ text: `${event.receiver} (drop)`, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.receiver && setTooltip({ text: `${event.receiver} (drop)`, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
        </g>
      )
    }

    // Handle throwaways (type 22) - use turnover coordinates
    if (isThrowaway && throwerX != null && throwerY != null && turnoverX != null && turnoverY != null) {
      // Normalize coordinates based on team direction
      const normThrowerX = normalizeX(throwerX, team)
      const normThrowerY = normalizeY(throwerY, team)
      const normTurnoverX = normalizeX(turnoverX, team)
      const normTurnoverY = normalizeY(turnoverY, team)

      const x1 = scaleX(normThrowerY)
      const y1 = scaleY(normThrowerX)
      const x2 = scaleX(normTurnoverY)
      const y2 = scaleY(normTurnoverX)

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
          {/* Throw line */}
          <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={color}
            strokeWidth="2"
            strokeDasharray="5,5"
            markerEnd="url(#arrowhead-red)"
          />
          {/* Thrower dot */}
          <circle
            cx={x1}
            cy={y1}
            r="4"
            fill={color}
            onMouseEnter={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => event.thrower && setTooltip({ text: event.thrower, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
          {/* Throwaway location - red square */}
          <rect
            x={x2 - 5}
            y={y2 - 5}
            width="10"
            height="10"
            fill={color}
            stroke="#fff"
            strokeWidth="1"
            onMouseEnter={(e) => setTooltip({ text: throwawayTooltip, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setTooltip({ text: throwawayTooltip, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTooltip(null)}
          />
        </g>
      )
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
      </div>

      {selectedPossession !== null && (
        <div className="selection-info">
          <span>Possession {selectedPossession + 1} of {possessions.length} selected</span>
          <button onClick={() => setSelectedPossession(null)} className="clear-btn">
            Clear Selection
          </button>
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

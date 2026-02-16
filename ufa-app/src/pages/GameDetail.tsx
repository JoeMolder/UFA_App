import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, Game, GameEvent } from '../api/client'
import FieldVisualization from '../components/FieldVisualization'
import ScoreTimeline from '../components/ScoreTimeline'
import '../styles/GameDetail.css'

interface Point {
  homeScore: number
  awayScore: number
  events: GameEvent[]
  startIndex: number
  endIndex: number
}

function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>()
  const navigate = useNavigate()
  const [game, setGame] = useState<Game | null>(null)
  const [events, setEvents] = useState<GameEvent[]>([])
  const [points, setPoints] = useState<Point[]>([])
  const [selectedPointIndex, setSelectedPointIndex] = useState(0)
  const [teamFilter, setTeamFilter] = useState<string>('both') // 'both', 'home', 'away'
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      if (!gameId) return

      try {
        setLoading(true)
        setError(null)

        const [gameData, eventsData] = await Promise.all([
          api.getGame(gameId),
          api.getGameEvents(gameId)
        ])

        setGame(gameData)
        setEvents(eventsData)

        // Parse events into points
        const parsedPoints = parseEventsIntoPoints(eventsData, gameData)
        setPoints(parsedPoints)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch game data')
        console.error('Error fetching game data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [gameId])

  const parseEventsIntoPoints = (events: GameEvent[], game: Game): Point[] => {
    const points: Point[] = []
    let currentHomeScore = 0
    let currentAwayScore = 0
    let pointStartIndex = 0

    events.forEach((event, index) => {
      // Event type 19 is a goal (based on actual data)
      if (event.event_type === 19) {
        const pointEvents = events.slice(pointStartIndex, index + 1)

        points.push({
          homeScore: currentHomeScore,
          awayScore: currentAwayScore,
          events: pointEvents,
          startIndex: pointStartIndex,
          endIndex: index
        })

        // Update score based on which team scored
        if (event.team === game.home_team_id) {
          currentHomeScore++
        } else {
          currentAwayScore++
        }

        pointStartIndex = index + 1
      }
    })

    // Add remaining events as final point if any
    if (pointStartIndex < events.length) {
      points.push({
        homeScore: currentHomeScore,
        awayScore: currentAwayScore,
        events: events.slice(pointStartIndex),
        startIndex: pointStartIndex,
        endIndex: events.length - 1
      })
    }

    return points
  }

  const currentPoint = points[selectedPointIndex]
  const filteredEvents = currentPoint?.events.filter(event => {
    if (teamFilter === 'both') return true
    if (teamFilter === 'home') return event.team === game?.home_team_id
    if (teamFilter === 'away') return event.team === game?.away_team_id
    return true
  }) || []

  if (loading) {
    return (
      <div className="game-detail">
        <p>Loading game data...</p>
      </div>
    )
  }

  if (error || !game) {
    return (
      <div className="game-detail">
        <p style={{ color: 'red' }}>Error: {error || 'Game not found'}</p>
        <button onClick={() => navigate('/')}>Back to Home</button>
      </div>
    )
  }

  return (
    <div className="game-detail">
      <div className="game-header">
        <button onClick={() => navigate('/')} className="back-button">← Back</button>
        <h1>
          {game.away_team_id} ({game.away_score}) @ {game.home_team_id} ({game.home_score})
        </h1>
        <p className="game-date">{game.game_date}</p>
      </div>

      <div className="controls">
        <div className="team-filter">
          <label>Show Throws: </label>
          <button
            className={teamFilter === 'both' ? 'active' : ''}
            onClick={() => setTeamFilter('both')}
          >
            Both Teams
          </button>
          <button
            className={teamFilter === 'home' ? 'active' : ''}
            onClick={() => setTeamFilter('home')}
          >
            {game.home_team_id}
          </button>
          <button
            className={teamFilter === 'away' ? 'active' : ''}
            onClick={() => setTeamFilter('away')}
          >
            {game.away_team_id}
          </button>
        </div>
      </div>

      {currentPoint && (
        <FieldVisualization
          events={filteredEvents}
          homeTeam={game.home_team_id}
          awayTeam={game.away_team_id}
        />
      )}

      <ScoreTimeline
        points={points}
        selectedIndex={selectedPointIndex}
        onSelectPoint={setSelectedPointIndex}
        homeTeam={game.home_team_id}
        awayTeam={game.away_team_id}
      />
    </div>
  )
}

export default GameDetail

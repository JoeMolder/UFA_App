import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, Game, GameEvent } from '../api/client'
import { teamLabel } from '../utils'
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
  const [points, setPoints] = useState<Point[]>([])
  const [selectedPointIndex, setSelectedPointIndex] = useState(0)
  const [teamFilter, setTeamFilter] = useState<string>('both')
  const [roster, setRoster] = useState<Record<string, { id: string; name: string; o_pts: number; d_pts: number }[]>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      if (!gameId) return

      try {
        setLoading(true)
        setError(null)

        const [gameData, eventsData, rosterData] = await Promise.all([
          api.getGame(gameId),
          api.getGameEvents(gameId),
          api.getGameRoster(gameId),
        ])

        setGame(gameData)
        setRoster(rosterData)

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
      // Event type 19 is a goal, 23 is a callahan (scores for the OTHER team)
      if (event.event_type === 19 || event.event_type === 23) {
        const pointEvents = events.slice(pointStartIndex, index + 1)

        points.push({
          homeScore: currentHomeScore,
          awayScore: currentAwayScore,
          events: pointEvents,
          startIndex: pointStartIndex,
          endIndex: index
        })

        // Update score based on which team scored
        if (event.event_type === 23) {
          // Callahan: the team field is the offense (who threw it), so the OTHER team scores
          if (event.team === game.home_team_id) {
            currentAwayScore++
          } else {
            currentHomeScore++
          }
        } else {
          // Regular goal
          if (event.team === game.home_team_id) {
            currentHomeScore++
          } else {
            currentAwayScore++
          }
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
          <span style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => navigate(`/team/${game.away_team_id}`)}>{teamLabel(game.away_team_id)}</span>
          {' '}({game.away_score}) @ <span style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => navigate(`/team/${game.home_team_id}`)}>{teamLabel(game.home_team_id)}</span>
          {' '}({game.home_score})
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
            {teamLabel(game.home_team_id)}
          </button>
          <button
            className={teamFilter === 'away' ? 'active' : ''}
            onClick={() => setTeamFilter('away')}
          >
            {teamLabel(game.away_team_id)}
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

      {Object.keys(roster).length > 0 && (
        <div style={{ marginTop: '32px' }}>
          <h2 style={{ marginBottom: '16px', fontSize: '18px' }}>Rosters</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            {[game.away_team_id, game.home_team_id].map(teamId => {
              const players = roster[teamId] || []
              return (
                <div key={teamId}>
                  <h3
                    style={{ fontSize: '15px', color: '#60a5fa', marginBottom: '8px', cursor: 'pointer' }}
                    onClick={() => navigate(`/team/${teamId}`)}
                  >
                    {teamLabel(teamId)}
                  </h3>
                  {players.length === 0 ? (
                    <p style={{ color: '#666', fontSize: '13px' }}>No lineup data</p>
                  ) : (
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid #444', color: '#888' }}>
                          <th style={{ textAlign: 'left', paddingBottom: '6px' }}>Player</th>
                          <th style={{ textAlign: 'center', paddingBottom: '6px', width: '40px' }}>Line</th>
                          <th style={{ textAlign: 'right', paddingBottom: '6px' }}>O pts</th>
                          <th style={{ textAlign: 'right', paddingBottom: '6px' }}>D pts</th>
                        </tr>
                      </thead>
                      <tbody>
                        {players.map(p => {
                          const line = p.o_pts >= p.d_pts ? 'O' : 'D'
                          return (
                            <tr key={p.id} style={{ borderBottom: '1px solid #2a2a3e' }}>
                              <td
                                style={{ padding: '5px 0', cursor: 'pointer', color: '#e2e8f0' }}
                                onClick={() => navigate(`/player/${p.id}`)}
                              >{p.name}</td>
                              <td style={{ textAlign: 'center', padding: '5px 0' }}>
                                <span style={{
                                  padding: '1px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 700,
                                  backgroundColor: line === 'O' ? 'rgba(99,102,241,0.2)' : 'rgba(239,68,68,0.2)',
                                  color: line === 'O' ? '#818cf8' : '#f87171',
                                }}>{line}</span>
                              </td>
                              <td style={{ textAlign: 'right', padding: '5px 0', color: '#aaa' }}>{p.o_pts}</td>
                              <td style={{ textAlign: 'right', padding: '5px 0', color: '#aaa' }}>{p.d_pts}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default GameDetail

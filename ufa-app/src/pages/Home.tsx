import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, Game, StatsResponse } from '../api/client'

function Home() {
  const navigate = useNavigate()
  const [games, setGames] = useState<Game[]>([])
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const [statsData, gamesData] = await Promise.all([
          api.getStats(),
          api.getGames(10)
        ])

        setStats(statsData)
        setGames(gamesData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
        console.error('Error fetching data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const handleGameClick = (gameId: string) => {
    navigate(`/game/${gameId}`)
  }

  if (loading) {
    return (
      <div className="app">
        <h1>UFA Analytics</h1>
        <p>Loading...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <h1>UFA Analytics</h1>
        <p style={{ color: 'red' }}>Error: {error}</p>
        <p>Make sure FastAPI is running at http://localhost:8000</p>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>🏆 UFA Analytics Platform</h1>
        <p>Ultimate Frisbee Association Data Analysis</p>
      </header>

      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={() => navigate('/predict')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#0ea5e9',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
        >
          Throw Prediction Heatmap
        </button>
        <button
          onClick={() => navigate('/embeddings')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#8b5cf6',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Player Embeddings
        </button>
      </div>

      {stats && (
        <div className="stats-summary">
          <h2>Database Summary</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{stats.total_games}</div>
              <div className="stat-label">Total Games</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.total_events.toLocaleString()}</div>
              <div className="stat-label">Total Events</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.synthetic_events}</div>
              <div className="stat-label">Synthetic Events</div>
            </div>
          </div>
        </div>
      )}

      <div className="games-section">
        <h2>Recent Games</h2>
        <div className="games-list">
          {games.map((game) => (
            <div
              key={game.game_id}
              className="game-card"
              onClick={() => handleGameClick(game.game_id)}
            >
              <div className="game-date">{game.game_date}</div>
              <div className="game-matchup">
                <div className="team">
                  <span className="team-name">{game.away_team_id}</span>
                  <span className="score">{game.away_score}</span>
                </div>
                <span className="vs">@</span>
                <div className="team">
                  <span className="team-name">{game.home_team_id}</span>
                  <span className="score">{game.home_score}</span>
                </div>
              </div>
              <div className="game-id">{game.game_id}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Home

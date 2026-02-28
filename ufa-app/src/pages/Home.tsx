import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, Game, StatsResponse } from '../api/client'

function Home() {
  const navigate = useNavigate()
  const [games, setGames] = useState<Game[]>([])
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const [statsData, gamesData] = await Promise.all([
          api.getStats(),
          api.getGames(20)
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

  useEffect(() => {
    const timeout = setTimeout(async () => {
      try {
        const results = await api.getGames(searchQuery ? 50 : 20, searchQuery)
        setGames(results)
      } catch (err) {
        console.error('Search error:', err)
      }
    }, 300)
    return () => clearTimeout(timeout)
  }, [searchQuery])

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
        <button
          onClick={() => navigate('/turnovers')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Turnover Heatmap
        </button>
        <button
          onClick={() => navigate('/pull-plays')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#14b8a6',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Pull Plays
        </button>
        <button
          onClick={() => navigate('/epv')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#f97316',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          EPV Heatmap
        </button>
        <button
          onClick={() => navigate('/zone-strategy')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#ec4899',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Zone Strategy Map
        </button>
        <button
          onClick={() => navigate('/completion')}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#22c55e',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Completion %
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
        <h2>Games</h2>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by team or date (e.g. empire, 2024-05)..."
          style={{
            width: '100%',
            maxWidth: '400px',
            padding: '8px 12px',
            fontSize: '14px',
            borderRadius: '6px',
            border: '1px solid #555',
            backgroundColor: '#2a2a3e',
            color: 'white',
            boxSizing: 'border-box',
            marginBottom: '16px',
          }}
        />
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

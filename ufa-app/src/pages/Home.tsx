import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, Game, StatsResponse, PlayerOption } from '../api/client'
import { teamLabel } from '../utils'

function Home() {
  const navigate = useNavigate()
  const [games, setGames] = useState<Game[]>([])
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [players, setPlayers] = useState<PlayerOption[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

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
    api.getSynergyPlayers().then(setPlayers).catch(console.error)
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) setShowDropdown(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
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

  const filteredPlayers = searchQuery.trim().length >= 2
    ? players.filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase())).slice(0, 6)
    : []

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
        <h1>UFA Analytics</h1>
        <p>Ultimate Frisbee Association · 2021–2025</p>
      </header>

      <div className="feature-grid">
        {[
          { label: 'Throw Prediction', desc: 'Heatmap of predicted throw success rates across the field', path: '/predict', icon: '🎯', color: '#0ea5e9' },
          { label: 'EPV Heatmap', desc: 'Expected possession value across field positions', path: '/epv', icon: '📊', color: '#f97316' },
          { label: 'Turnover Heatmap', desc: 'Spatial analysis of where turnovers occur and are predicted', path: '/turnovers', icon: '🔴', color: '#ef4444' },
{ label: 'Line Synergy', desc: 'Pair synergy analysis and lineup scoring prediction', path: '/line-synergy', icon: '🤝', color: '#6366f1' },
          { label: 'Player Embeddings', desc: 'Visualize player similarity and clustering in 2D space', path: '/embeddings', icon: '🧬', color: '#8b5cf6' },
        ].map(({ label, desc, path, icon, color }) => (
          <div
            key={path}
            className="feature-card"
            onClick={() => navigate(path)}
            style={{ '--card-color': color } as React.CSSProperties}
          >
            <div className="feature-icon">{icon}</div>
            <div className="feature-name">{label}</div>
            <div className="feature-desc">{desc}</div>
          </div>
        ))}
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
        <div ref={searchRef} style={{ position: 'relative', maxWidth: '400px', marginBottom: '16px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setShowDropdown(true) }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search team, date, or player name..."
            style={{
              width: '100%',
              padding: '8px 12px',
              fontSize: '14px',
              borderRadius: '6px',
              border: '1px solid #555',
              backgroundColor: '#2a2a3e',
              color: 'white',
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
          {showDropdown && filteredPlayers.length > 0 && (
            <div style={{
              position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
              background: '#1e1e2e', border: '1px solid #444', borderRadius: '6px',
              zIndex: 100, overflow: 'hidden',
            }}>
              <div style={{ padding: '5px 12px 3px', fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Players</div>
              {filteredPlayers.map((p) => (
                <div
                  key={p.id}
                  onClick={() => { setShowDropdown(false); navigate(`/player/${p.id}`) }}
                  style={{ padding: '8px 12px', cursor: 'pointer', fontSize: '14px', borderTop: '1px solid #2a2a3e' }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2a3e')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  {p.name}
                </div>
              ))}
            </div>
          )}
        </div>
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
                  <span
                    className="team-name"
                    onClick={(e) => { e.stopPropagation(); navigate(`/team/${game.away_team_id}`); }}
                    style={{ cursor: 'pointer', textDecoration: 'underline' }}
                  >{teamLabel(game.away_team_id)}</span>
                  <span className="score">{game.away_score}</span>
                </div>
                <span className="vs">@</span>
                <div className="team">
                  <span
                    className="team-name"
                    onClick={(e) => { e.stopPropagation(); navigate(`/team/${game.home_team_id}`); }}
                    style={{ cursor: 'pointer', textDecoration: 'underline' }}
                  >{teamLabel(game.home_team_id)}</span>
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

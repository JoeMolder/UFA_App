import { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { api, PlayerOption } from '../api/client'

const NAV_LINKS = [
  { label: 'Predict', path: '/predict' },
  { label: 'EPV', path: '/epv' },
  { label: 'Turnovers', path: '/turnovers' },
  { label: 'Completion', path: '/completion' },
  { label: 'Synergy', path: '/line-synergy' },
  { label: 'Embeddings', path: '/embeddings' },
]

export function NavBar() {
  const navigate = useNavigate()
  const location = useLocation()
  const [query, setQuery] = useState('')
  const [players, setPlayers] = useState<PlayerOption[]>([])
  const [teams, setTeams] = useState<string[]>([])
  const [open, setOpen] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.getSynergyPlayers().then(setPlayers).catch(console.error)
    api.getTeams().then(setTeams).catch(console.error)
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const q = query.toLowerCase().trim()
  const filteredTeams = q.length >= 1 ? teams.filter((t) => t.toLowerCase().includes(q)).slice(0, 4) : []
  const filteredPlayers = q.length >= 2 ? players.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 5) : []
  const hasResults = filteredTeams.length > 0 || filteredPlayers.length > 0

  const handleSelect = (path: string) => {
    setQuery('')
    setOpen(false)
    navigate(path)
  }

  return (
    <nav className="navbar">
      <div className="navbar-brand" onClick={() => navigate('/')}>
        UFA Analytics
      </div>
      <div className="navbar-links">
        {NAV_LINKS.map(({ label, path }) => (
          <button
            key={path}
            className={`nav-link${location.pathname === path ? ' active' : ''}`}
            onClick={() => navigate(path)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="navbar-search" ref={searchRef}>
        <input
          className="navbar-search-input"
          type="text"
          placeholder="Search players & teams..."
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
        />
        {open && hasResults && (
          <div className="navbar-search-dropdown">
            {filteredTeams.length > 0 && (
              <>
                <div className="search-section-label">Teams</div>
                {filteredTeams.map((t) => (
                  <div key={t} className="search-result-item" onClick={() => handleSelect(`/team/${t}`)}>
                    {t}
                  </div>
                ))}
              </>
            )}
            {filteredPlayers.length > 0 && (
              <>
                <div className="search-section-label">Players</div>
                {filteredPlayers.map((p) => (
                  <div key={p.id} className="search-result-item" onClick={() => handleSelect(`/player/${p.id}`)}>
                    {p.name}
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </nav>
  )
}

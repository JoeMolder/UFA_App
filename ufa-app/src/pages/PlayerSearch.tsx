import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, PlayerOption } from '../api/client'

function PlayerSearch() {
  const navigate = useNavigate()
  const [players, setPlayers] = useState<PlayerOption[]>([])
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.getSynergyPlayers().then(setPlayers).catch(console.error)
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = query
    ? players.filter((p) => p.name.toLowerCase().includes(query.toLowerCase())).slice(0, 50)
    : players.slice(0, 50)

  return (
    <div className="app">
      <button
        onClick={() => navigate('/')}
        style={{ marginBottom: '16px', padding: '6px 14px', cursor: 'pointer', borderRadius: '6px', border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white' }}
      >
        ← Back
      </button>

      <header style={{ marginBottom: '32px' }}>
        <h1>Player Lookup</h1>
        <p style={{ color: '#aaa', fontSize: '14px' }}>Search for a player to view their career stats</p>
      </header>

      <div ref={ref} style={{ position: 'relative', maxWidth: '360px' }}>
        <input
          type="text"
          placeholder="Type a player name..."
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
          style={{
            width: '100%', padding: '10px 14px', background: '#1e1e2e',
            border: '1px solid #444', borderRadius: '6px', color: '#fff',
            fontSize: '15px', boxSizing: 'border-box', outline: 'none',
          }}
        />
        {open && filtered.length > 0 && (
          <div style={{
            position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
            background: '#1e1e2e', border: '1px solid #444', borderRadius: '6px',
            maxHeight: '280px', overflowY: 'auto', zIndex: 100,
          }}>
            {filtered.map((p) => (
              <div
                key={p.id}
                onClick={() => navigate(`/player/${p.id}`)}
                style={{
                  padding: '9px 14px', cursor: 'pointer', fontSize: '14px',
                  borderBottom: '1px solid #2a2a3e',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2a3e')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                {p.name}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default PlayerSearch

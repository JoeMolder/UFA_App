import { useEffect, useState, useCallback, useRef } from 'react'
import { api, EmbeddingsResponse } from '../api/client'
import PlayerEmbeddings from '../components/PlayerEmbeddings'

function Embeddings() {
  const [embeddingsData, setEmbeddingsData] = useState<EmbeddingsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const fetchEmbeddings = async () => {
      try {
        const data = await api.getEmbeddings()
        setEmbeddingsData(data)
      } catch (err) {
        setError('Failed to load embeddings. Is the backend running?')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchEmbeddings()
  }, [])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handlePlayerSelect = useCallback((playerId: string) => {
    setSelectedPlayer(playerId)
    setShowDropdown(false)
  }, [])

  // Search by full name, show dropdown only when typing
  const filteredPlayers = embeddingsData && searchQuery.trim().length >= 2
    ? embeddingsData.players.filter((id) => {
        const fullName = embeddingsData.name_map[id] ?? id
        return fullName.toLowerCase().includes(searchQuery.toLowerCase())
      })
    : []

  const displayName = selectedPlayer && embeddingsData
    ? (embeddingsData.name_map[selectedPlayer] ?? selectedPlayer)
    : null

  if (loading) {
    return <div style={{ padding: '40px', textAlign: 'center', color: 'white' }}>Loading embeddings...</div>
  }

  if (error || !embeddingsData) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#ef4444' }}><p>{error}</p></div>
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>Player Embeddings</h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        UMAP projection of player throwing style. Players close together have similar throwing patterns.{' '}
        {embeddingsData.players.length} players.
      </p>

      {/* Player search */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
        <div ref={searchRef} style={{ position: 'relative', width: '300px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setShowDropdown(true) }}
            onFocus={() => { if (searchQuery.trim().length >= 2) setShowDropdown(true) }}
            placeholder="Search players..."
            style={{
              width: '100%', padding: '8px 12px', fontSize: '14px', borderRadius: '6px',
              border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white',
              boxSizing: 'border-box', outline: 'none',
            }}
          />
          {showDropdown && filteredPlayers.length > 0 && (
            <div style={{
              position: 'absolute', top: '100%', left: 0, right: 0,
              maxHeight: '220px', overflowY: 'auto',
              backgroundColor: '#1e1e2e', border: '1px solid #444',
              borderRadius: '0 0 6px 6px', zIndex: 100,
            }}>
              {filteredPlayers.slice(0, 20).map((id) => {
                const name = embeddingsData.name_map[id] ?? id
                return (
                  <div
                    key={id}
                    onClick={() => { handlePlayerSelect(id); setSearchQuery(name) }}
                    style={{
                      padding: '7px 12px', cursor: 'pointer', fontSize: '13px',
                      color: id === selectedPlayer ? '#818cf8' : '#ccc',
                      borderTop: '1px solid #2a2a3e',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2a3e')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {name}
                  </div>
                )
              })}
            </div>
          )}
        </div>
        {displayName && (
          <span style={{ color: '#818cf8', fontSize: '14px' }}>{displayName}</span>
        )}
      </div>

      <PlayerEmbeddings
        players={embeddingsData.players}
        coordinates={embeddingsData.coordinates}
        clusters={embeddingsData.clusters}
        playerStats={embeddingsData.player_stats}
        nameMap={embeddingsData.name_map}
        selectedPlayer={selectedPlayer}
        onPlayerClick={handlePlayerSelect}
      />

      <div style={{ color: '#888', fontSize: '13px', marginTop: '12px' }}>
        Scroll to zoom, drag to pan. Click a dot or search to focus on a player.
      </div>
    </div>
  )
}

export default Embeddings

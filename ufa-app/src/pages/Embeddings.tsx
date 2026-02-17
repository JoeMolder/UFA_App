import { useEffect, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, EmbeddingsResponse } from '../api/client'
import PlayerEmbeddings from '../components/PlayerEmbeddings'

function Embeddings() {
  const [embeddingsData, setEmbeddingsData] = useState<EmbeddingsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)

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

  const handlePlayerSelect = useCallback((player: string) => {
    setSelectedPlayer(player)
    setSearchQuery(player)
    setShowDropdown(false)
  }, [])

  const filteredPlayers = embeddingsData
    ? embeddingsData.players.filter((p) =>
        p.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : []

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'white' }}>
        Loading embeddings...
      </div>
    )
  }

  if (error || !embeddingsData) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#ef4444' }}>
        <p>{error}</p>
        <Link to="/" style={{ color: 'cyan' }}>Back to Home</Link>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>
        Player Embeddings
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        UMAP projection of player throwing style embeddings. Players close together have similar
        throwing patterns. {embeddingsData.players.length} players.
      </p>

      {/* Player search */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
        <div style={{ position: 'relative', width: '300px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              setShowDropdown(true)
            }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search players..."
            style={{
              width: '100%',
              padding: '8px 12px',
              fontSize: '14px',
              borderRadius: '6px',
              border: '1px solid #555',
              backgroundColor: '#2a2a3e',
              color: 'white',
              boxSizing: 'border-box',
            }}
          />
          {showDropdown && filteredPlayers.length > 0 && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                maxHeight: '200px',
                overflowY: 'auto',
                backgroundColor: '#2a2a3e',
                border: '1px solid #555',
                borderRadius: '0 0 6px 6px',
                zIndex: 10,
              }}
            >
              {filteredPlayers.slice(0, 20).map((p) => (
                <div
                  key={p}
                  onClick={() => handlePlayerSelect(p)}
                  style={{
                    padding: '6px 12px',
                    cursor: 'pointer',
                    color: p === selectedPlayer ? 'cyan' : 'white',
                    backgroundColor: p === selectedPlayer ? '#3a3a5e' : 'transparent',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#3a3a5e')}
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor =
                      p === selectedPlayer ? '#3a3a5e' : 'transparent')
                  }
                >
                  {p}
                </div>
              ))}
            </div>
          )}
        </div>
        {selectedPlayer && (
          <span style={{ color: 'cyan', fontSize: '14px', fontFamily: 'monospace' }}>
            {selectedPlayer}
          </span>
        )}
      </div>

      <PlayerEmbeddings
        players={embeddingsData.players}
        coordinates={embeddingsData.coordinates}
        clusters={embeddingsData.clusters}
        playerStats={embeddingsData.player_stats}
        clusterSummaries={embeddingsData.cluster_summaries}
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

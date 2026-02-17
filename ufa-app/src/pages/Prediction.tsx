import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import ThrowHeatmap from '../components/ThrowHeatmap'

function Prediction() {
  const [players, setPlayers] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const data = await api.getPlayers()
        setPlayers(data)
      } catch (err) {
        setError('Failed to load players. Is the backend running?')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchPlayers()
  }, [])

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'white' }}>
        Loading model...
      </div>
    )
  }

  if (error) {
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
        Throw Prediction Heatmap
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        Select a player and click/drag on the field to see predicted throw distributions.
        {' '}{players.length} players available.
      </p>

      <ThrowHeatmap players={players} />
    </div>
  )
}

export default Prediction

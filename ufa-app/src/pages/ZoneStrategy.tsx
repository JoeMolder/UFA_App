import { Link } from 'react-router-dom'
import ZoneStrategyMap from '../components/ZoneStrategyMap'

function ZoneStrategy() {
  return (
    <div style={{ padding: '20px', maxWidth: '1060px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>
        Zone Strategy Map
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        The field is divided into 12 zones. For each zone, the arrows show the average first 3 throws
        from any possession start (pulls + turnovers) originating there. Zone brightness indicates
        how many possessions start in that zone. Filter by team to see team-specific tendencies.
      </p>

      <ZoneStrategyMap />
    </div>
  )
}

export default ZoneStrategy

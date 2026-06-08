import { Link } from 'react-router-dom'
import EPVHeatmap from '../components/EPVHeatmap'

function EPV() {
  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>
        Expected Possession Value (EPV)
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        P(score before turnover) for each field position. Use the throw index slider to see how
        scoring probability shifts as a possession progresses. Filter by team for team-specific EPV.
        All coordinates are direction-normalized so teams attack toward Y=120.
      </p>

      <EPVHeatmap />
    </div>
  )
}

export default EPV

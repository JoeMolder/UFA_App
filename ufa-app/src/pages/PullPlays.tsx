import { Link } from 'react-router-dom'
import PullPlayVisualizer from '../components/PullPlayVisualizer'

function PullPlays() {
  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>
        Pull Play Sequences
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        Expected first 3 throws after a pull based on where the disc lands. Drag the dot to change
        the pull landing position. All coordinates are direction-normalized so teams attack toward y=120.
      </p>

      <PullPlayVisualizer />
    </div>
  )
}

export default PullPlays

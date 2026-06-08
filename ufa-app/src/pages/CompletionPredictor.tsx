import { Link } from 'react-router-dom'
import CompletionMap from '../components/CompletionMap'

function CompletionPredictor() {
  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '8px', fontSize: '24px' }}>
        Throw Completion Predictor
      </h1>
      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        Select a thrower, click the field to set a throw origin (cyan dot), then click a target
        position to see that player's completion probability for the exact throw. A heatmap of
        completion % across all targets is shown immediately after setting the origin.
        All coordinates are direction-normalized so offenses attack toward Y=120.
      </p>

      <CompletionMap />
    </div>
  )
}

export default CompletionPredictor

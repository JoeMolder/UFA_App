import { useState } from 'react'
import { Link } from 'react-router-dom'
import TurnoverHeatmap from '../components/TurnoverHeatmap'
import TurnoverRateHeatmap from '../components/TurnoverRateHeatmap'
import TurnoverOriginsHeatmap from '../components/TurnoverOriginsHeatmap'

type Tab = 'prediction' | 'blocks' | 'origins'

const tabs: { key: Tab; label: string; description: string }[] = [
  {
    key: 'prediction',
    label: 'Turnovers',
    description: 'Predicted turnover destinations from any field position. Drag the dot to see where turnovers are likely to land.',
  },
  {
    key: 'blocks',
    label: 'Blocks',
    description: 'Predicted block destinations from any field position. Drag the dot to see where blocks are likely to land.',
  },
  {
    key: 'origins',
    label: 'Turnover Origins',
    description: 'Turnover rate by field position — what percentage of throws from each spot result in a turnover? Green = low, red = high.',
  },
]

function Turnovers() {
  const [activeTab, setActiveTab] = useState<Tab>('prediction')

  const current = tabs.find(t => t.key === activeTab)!

  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <Link to="/" style={{ color: 'cyan', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Home
        </Link>
      </div>

      <h1 style={{ color: 'white', marginBottom: '16px', fontSize: '24px' }}>
        Turnover Heatmaps
      </h1>

      <div style={{ display: 'flex', gap: '4px', marginBottom: '16px' }}>
        {tabs.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '8px 20px',
              fontSize: '14px',
              fontFamily: 'monospace',
              border: 'none',
              borderRadius: '6px 6px 0 0',
              cursor: 'pointer',
              background: activeTab === tab.key ? '#2a5934' : '#1a1a2e',
              color: activeTab === tab.key ? 'white' : '#888',
              borderBottom: activeTab === tab.key ? '2px solid cyan' : '2px solid transparent',
              transition: 'all 0.15s',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <p style={{ color: '#888', marginBottom: '24px', fontSize: '14px' }}>
        {current.description}
      </p>

      {activeTab === 'prediction' && <TurnoverHeatmap />}
      {activeTab === 'blocks' && <TurnoverRateHeatmap />}
      {activeTab === 'origins' && <TurnoverOriginsHeatmap />}
    </div>
  )
}

export default Turnovers

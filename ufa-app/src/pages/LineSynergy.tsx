import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, PlayerOption, SynergyPairResponse, LineupPredictResponse } from '../api/client'

// Searchable player dropdown
function PlayerSelect({
  label,
  players,
  value,
  onChange,
}: {
  label: string
  players: PlayerOption[]
  value: string
  onChange: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const selected = players.find((p) => p.id === value)
  const filtered = query
    ? players.filter((p) => p.name.toLowerCase().includes(query.toLowerCase())).slice(0, 50)
    : players.slice(0, 50)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={ref} style={{ position: 'relative', minWidth: '200px' }}>
      <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '4px' }}>{label}</div>
      <div
        onClick={() => setOpen((o) => !o)}
        style={{
          padding: '8px 10px',
          background: '#1e1e2e',
          border: '1px solid #444',
          borderRadius: '6px',
          cursor: 'pointer',
          color: selected ? '#fff' : '#888',
          fontSize: '14px',
          userSelect: 'none',
        }}
      >
        {selected ? selected.name : 'Select player...'}
      </div>
      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            zIndex: 100,
            background: '#1e1e2e',
            border: '1px solid #444',
            borderRadius: '6px',
            maxHeight: '260px',
            overflow: 'auto',
            marginTop: '2px',
          }}
        >
          <input
            autoFocus
            placeholder="Search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              padding: '8px 10px',
              background: '#16213e',
              border: 'none',
              borderBottom: '1px solid #333',
              color: '#fff',
              fontSize: '14px',
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
          {filtered.map((p) => (
            <div
              key={p.id}
              onClick={() => { onChange(p.id); setQuery(''); setOpen(false) }}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                fontSize: '14px',
                color: p.id === value ? '#6366f1' : '#ddd',
                background: p.id === value ? '#2a2a4e' : 'transparent',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a3e')}
              onMouseLeave={(e) => (e.currentTarget.style.background = p.id === value ? '#2a2a4e' : 'transparent')}
            >
              {p.name}
            </div>
          ))}
          {filtered.length === 0 && (
            <div style={{ padding: '10px 12px', color: '#888', fontSize: '14px' }}>No results</div>
          )}
        </div>
      )}
    </div>
  )
}

function SynergyBadge({ delta }: { delta: number }) {
  const pct = (delta * 100).toFixed(1)
  const color = delta > 0.05 ? '#22c55e' : delta < -0.05 ? '#ef4444' : '#f59e0b'
  const sign = delta > 0 ? '+' : ''
  return (
    <div style={{
      display: 'inline-block',
      padding: '6px 16px',
      borderRadius: '999px',
      background: color + '22',
      border: `2px solid ${color}`,
      color,
      fontSize: '22px',
      fontWeight: 700,
    }}>
      {sign}{pct} pp
    </div>
  )
}

export default function LineSynergy() {
  const navigate = useNavigate()
  const [players, setPlayers] = useState<PlayerOption[]>([])
  const [loading, setLoading] = useState(true)

  // Pair synergy state
  const [p1, setP1] = useState('')
  const [p2, setP2] = useState('')
  const [synergyResult, setSynergyResult] = useState<SynergyPairResponse | null>(null)
  const [synergyLoading, setSynergyLoading] = useState(false)
  const [synergyError, setSynergyError] = useState('')

  // Lineup predictor state
  const [lineup, setLineup] = useState<string[]>(['', '', '', '', '', '', ''])
  const [lineupResult, setLineupResult] = useState<LineupPredictResponse | null>(null)
  const [lineupLoading, setLineupLoading] = useState(false)
  const [lineupError, setLineupError] = useState('')

  useEffect(() => {
    api.getSynergyPlayers().then((data) => {
      setPlayers(data)
      setLoading(false)
    })
  }, [])

  const handleSynergy = async () => {
    if (!p1 || !p2 || p1 === p2) return
    setSynergyLoading(true)
    setSynergyError('')
    setSynergyResult(null)
    try {
      const result = await api.getSynergyPair(p1, p2)
      setSynergyResult(result)
    } catch (e: any) {
      setSynergyError(e?.response?.data?.detail || 'Error fetching synergy data')
    } finally {
      setSynergyLoading(false)
    }
  }

  const handleLineup = async () => {
    if (lineup.some((p) => !p)) return
    setLineupLoading(true)
    setLineupError('')
    setLineupResult(null)
    try {
      const result = await api.getLineupPredict(lineup)
      setLineupResult(result)
    } catch (e: any) {
      setLineupError(e?.response?.data?.detail || 'Error predicting lineup')
    } finally {
      setLineupLoading(false)
    }
  }

  const probColor = (p: number) => p >= 0.65 ? '#22c55e' : p >= 0.55 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: '30px', maxWidth: '900px', margin: '0 auto', fontFamily: 'sans-serif', color: '#e0e0e0' }}>
      <button
        onClick={() => navigate('/')}
        style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', marginBottom: '20px', fontSize: '14px' }}
      >
        ← Back
      </button>

      <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '6px' }}>Line Synergy</h1>
      <p style={{ color: '#888', marginBottom: '36px' }}>
        Analyze how players work together and predict O-line scoring chances.
      </p>

      {/* ── Section A: Pair Synergy ─────────────────────────────── */}
      <div style={{ background: '#1a1a2e', borderRadius: '12px', padding: '24px', marginBottom: '28px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px', color: '#a5b4fc' }}>
          Pair Synergy
        </h2>

        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: '16px' }}>
          {loading ? (
            <div style={{ color: '#888' }}>Loading players...</div>
          ) : (
            <>
              <PlayerSelect label="Player 1" players={players} value={p1} onChange={setP1} />
              <PlayerSelect label="Player 2" players={players} value={p2} onChange={setP2} />
              <button
                onClick={handleSynergy}
                disabled={!p1 || !p2 || p1 === p2 || synergyLoading}
                style={{
                  padding: '8px 20px',
                  background: '#6366f1',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  opacity: (!p1 || !p2 || p1 === p2) ? 0.5 : 1,
                  alignSelf: 'flex-end',
                  marginTop: '20px',
                }}
              >
                {synergyLoading ? 'Loading...' : 'Compare'}
              </button>
            </>
          )}
        </div>

        {synergyError && <div style={{ color: '#f87171', fontSize: '14px' }}>{synergyError}</div>}

        {synergyResult && (
          <div style={{ marginTop: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px', flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: '13px', color: '#888', marginBottom: '4px' }}>Synergy Delta</div>
                <SynergyBadge delta={synergyResult.synergy_delta} />
              </div>
              <div style={{ color: '#555', fontSize: '20px' }}>|</div>
              <div>
                <div style={{ fontSize: '13px', color: '#888', marginBottom: '4px' }}>Shared Possessions</div>
                <div style={{ fontSize: '20px', fontWeight: 600 }}>{synergyResult.shared_possessions}</div>
              </div>
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', marginBottom: '16px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #333' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#888', fontWeight: 500 }}>Metric</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px', color: '#a5b4fc', fontWeight: 500 }}>{synergyResult.player1.name}</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px', color: '#a5b4fc', fontWeight: 500 }}>{synergyResult.player2.name}</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px', color: '#888', fontWeight: 500 }}>Together</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid #222' }}>
                  <td style={{ padding: '8px 12px', color: '#ccc' }}>O-line Scoring Rate</td>
                  <td style={{ textAlign: 'right', padding: '8px 12px' }}>{(synergyResult.p1_scoring_rate * 100).toFixed(1)}%</td>
                  <td style={{ textAlign: 'right', padding: '8px 12px' }}>{(synergyResult.p2_scoring_rate * 100).toFixed(1)}%</td>
                  <td style={{ textAlign: 'right', padding: '8px 12px', fontWeight: 600 }}>{(synergyResult.combined_scoring_rate * 100).toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>

            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              {[
                { label: `${synergyResult.player1.name} → ${synergyResult.player2.name}`, data: synergyResult.p1_to_p2 },
                { label: `${synergyResult.player2.name} → ${synergyResult.player1.name}`, data: synergyResult.p2_to_p1 },
              ].map(({ label, data }) => (
                <div key={label} style={{ background: '#16213e', borderRadius: '8px', padding: '12px 16px', flex: '1', minWidth: '180px' }}>
                  <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>{label}</div>
                  <div style={{ fontSize: '20px', fontWeight: 700 }}>{data.count} <span style={{ fontSize: '13px', color: '#888', fontWeight: 400 }}>throws</span></div>
                  <div style={{ fontSize: '14px', color: '#a5b4fc' }}>{(data.completion_pct * 100).toFixed(1)}% completion</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Section B: Line Predictor ───────────────────────────── */}
      <div style={{ background: '#1a1a2e', borderRadius: '12px', padding: '24px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px', color: '#a5b4fc' }}>
          Line Scoring Predictor
        </h2>
        <p style={{ color: '#888', fontSize: '13px', marginBottom: '16px' }}>
          Select 7 players for an O-line and predict their scoring probability.
        </p>

        {loading ? (
          <div style={{ color: '#888' }}>Loading players...</div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
              {lineup.map((pid, i) => (
                <PlayerSelect
                  key={i}
                  label={`Player ${i + 1}`}
                  players={players}
                  value={pid}
                  onChange={(id) => {
                    const next = [...lineup]
                    next[i] = id
                    setLineup(next)
                  }}
                />
              ))}
            </div>

            <button
              onClick={handleLineup}
              disabled={lineup.some((p) => !p) || lineupLoading}
              style={{
                padding: '10px 24px',
                background: '#6366f1',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '15px',
                opacity: lineup.some((p) => !p) ? 0.5 : 1,
              }}
            >
              {lineupLoading ? 'Predicting...' : 'Predict Scoring Chance'}
            </button>
          </>
        )}

        {lineupError && <div style={{ color: '#f87171', fontSize: '14px', marginTop: '12px' }}>{lineupError}</div>}

        {lineupResult && (
          <div style={{ marginTop: '24px', textAlign: 'center' }}>
            <div style={{ fontSize: '14px', color: '#888', marginBottom: '8px' }}>Predicted Scoring Probability</div>
            <div style={{ fontSize: '64px', fontWeight: 800, color: probColor(lineupResult.probability), lineHeight: 1 }}>
              {(lineupResult.probability * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '13px', color: '#666', marginTop: '8px' }}>
              Based on {lineupResult.known_players} of 7 players with stats data
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

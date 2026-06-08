import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, PlayerResponse, PlayerSeason, ThrowTendencies } from '../api/client'
import { ThrowTendencyChart } from '../components/ThrowTendencyChart'
import { teamLabel } from '../utils'

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function dist(v: number) { return v > 0 ? `${v.toFixed(1)}` : '—' }

const synergyColor = (delta: number) => {
  if (delta > 0.05) return '#22c55e'
  if (delta < -0.05) return '#ef4444'
  return '#f59e0b'
}

function StatsTable({ rows, career }: { rows: PlayerSeason[]; career: PlayerSeason }) {
  const navigate = useNavigate()
  const cols: { key: string; label: string; fmt: (s: PlayerSeason) => React.ReactNode; bold: boolean }[] = [
    { key: 'year', label: 'Year', fmt: (s: PlayerSeason) => s.year === 0 ? 'Career' : String(s.year), bold: false },
    {
      key: 'team', label: 'Team', bold: false,
      fmt: (s: PlayerSeason) => s.team
        ? <span style={{ cursor: 'pointer', color: '#60a5fa' }} onClick={() => navigate(`/team/${s.team}`)}>{teamLabel(s.team)}</span>
        : '—',
    },
    { key: 'o_possessions', label: 'OPP', fmt: (s: PlayerSeason) => String(s.o_possessions), bold: false },
    { key: 'd_possessions', label: 'DPP', fmt: (s: PlayerSeason) => String(s.d_possessions), bold: false },
    { key: 'throw_attempts', label: 'Throws', fmt: (s: PlayerSeason) => String(s.throw_attempts), bold: false },
    { key: 'completions', label: 'Comp', fmt: (s: PlayerSeason) => String(s.completions), bold: false },
    { key: 'completion_pct', label: 'Comp%', fmt: (s: PlayerSeason) => pct(s.completion_pct), bold: true },
    { key: 'assists', label: 'Ast', fmt: (s: PlayerSeason) => String(s.assists), bold: false },
    { key: 'goals', label: 'Gls', fmt: (s: PlayerSeason) => String(s.goals), bold: false },
    { key: 'catches', label: 'Catches', fmt: (s: PlayerSeason) => String(s.catches), bold: false },
    { key: 'turnovers', label: 'TO', fmt: (s: PlayerSeason) => String(s.turnovers), bold: false },
    { key: 'huck_attempts', label: 'Hucks', fmt: (s: PlayerSeason) => String(s.huck_attempts), bold: false },
    { key: 'huck_pct', label: 'Huck%', fmt: (s: PlayerSeason) => s.huck_attempts > 0 ? pct(s.huck_pct) : '—', bold: true },
    { key: 'avg_throw_dist', label: 'Avg Dist', fmt: (s: PlayerSeason) => dist(s.avg_throw_dist), bold: false },
    { key: 'avg_throw_depth', label: 'Depth', fmt: (s: PlayerSeason) => dist(s.avg_throw_depth), bold: false },
    { key: 'o_hold_rate', label: 'O Hold%', fmt: (s: PlayerSeason) => s.o_possessions > 0 ? pct(s.o_hold_rate) : '—', bold: true },
  ]

  const thStyle: React.CSSProperties = { textAlign: 'right', paddingBottom: '8px', color: '#aaa', whiteSpace: 'nowrap', paddingRight: '12px' }
  const tdStyle: React.CSSProperties = { textAlign: 'right', padding: '7px 12px 7px 0', whiteSpace: 'nowrap' }
  const tdLeft: React.CSSProperties = { textAlign: 'left', padding: '7px 12px 7px 0', whiteSpace: 'nowrap' }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: '13px', minWidth: '800px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #444' }}>
            {cols.map((c) => (
              <th key={c.key} style={c.key === 'year' || c.key === 'team' ? { ...thStyle, textAlign: 'left' } : thStyle}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.year} style={{ borderBottom: '1px solid #2a2a3e' }}>
              {cols.map((c) => (
                <td key={c.key} style={c.key === 'year' || c.key === 'team' ? tdLeft : { ...tdStyle, fontWeight: c.bold ? 600 : 400 }}>
                  {c.fmt(s)}
                </td>
              ))}
            </tr>
          ))}
          {/* Career row */}
          <tr style={{ borderTop: '2px solid #555', backgroundColor: '#1e1e2e' }}>
            {cols.map((c) => (
              <td key={c.key} style={c.key === 'year' || c.key === 'team'
                ? { ...tdLeft, fontWeight: 700, color: '#e2e8f0' }
                : { ...tdStyle, fontWeight: c.bold ? 700 : 600, color: '#e2e8f0' }
              }>
                {c.fmt(career)}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function PlayerPage() {
  const { playerId } = useParams<{ playerId: string }>()
  const navigate = useNavigate()
  const [player, setPlayer] = useState<PlayerResponse | null>(null)
  const [tendencies, setTendencies] = useState<ThrowTendencies | null>(null)
  const [selectedYear, setSelectedYear] = useState<number | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!playerId) return
    const fetch = async () => {
      try {
        setLoading(true)
        setError(null)
        const [data, tend] = await Promise.all([
          api.getPlayer(playerId, selectedYear),
          api.getPlayerThrowTendencies(playerId, selectedYear),
        ])
        setPlayer(data)
        setTendencies(tend)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch player data')
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [playerId, selectedYear])

  if (loading) return <div className="app"><p>Loading player data...</p></div>
  if (error) return <div className="app"><p style={{ color: 'red' }}>Error: {error}</p></div>
  if (!player) return null

  const btnBase: React.CSSProperties = {
    padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '13px',
    color: 'white', border: '1px solid #555',
  }

  return (
    <div className="app">
      <button
        onClick={() => navigate(-1)}
        style={{ marginBottom: '16px', padding: '6px 14px', cursor: 'pointer', borderRadius: '6px', border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white' }}
      >
        ← Back
      </button>

      <header style={{ marginBottom: '16px' }}>
        <h1>{player.player.name}</h1>
        <p style={{ color: '#aaa', fontSize: '14px' }}>{player.player.id}</p>
      </header>

      {/* Season selector */}
      {player.available_years.length > 1 && (
        <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '14px', color: '#aaa' }}>Season:</span>
          <button onClick={() => setSelectedYear(undefined)} style={{ ...btnBase, backgroundColor: selectedYear === undefined ? '#8b5cf6' : '#2a2a3e' }}>All</button>
          {player.available_years.map((y) => (
            <button key={y} onClick={() => setSelectedYear(y)} style={{ ...btnBase, backgroundColor: selectedYear === y ? '#8b5cf6' : '#2a2a3e' }}>
              {y}
            </button>
          ))}
        </div>
      )}

      {/* Stats table */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ marginBottom: '12px' }}>Season Stats</h2>
        {player.seasons.length === 0 ? (
          <p style={{ color: '#888', fontSize: '14px' }}>No data for this season.</p>
        ) : (
          <StatsTable rows={player.seasons} career={player.career} />
        )}
      </div>

      {/* Throw tendency chart */}
      {tendencies && tendencies.total_throws > 0 && (
        <div style={{ marginBottom: '32px' }}>
          <h2 style={{ marginBottom: '4px' }}>Throw Tendencies</h2>
          <p style={{ color: '#888', fontSize: '12px', marginBottom: '16px' }}>
            Sector length = avg distance · color = frequency (blue → purple)
          </p>
          <ThrowTendencyChart
            bins={tendencies.bins}
            totalThrows={tendencies.total_throws}
            maxAvgDist={tendencies.max_avg_dist}
          />
        </div>
      )}

      {/* Top connections */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        <div>
          <h2 style={{ marginBottom: '12px' }}>Top Targets</h2>
          <ConnectionTable rows={player.top_targets} navigate={navigate} />
        </div>
        <div>
          <h2 style={{ marginBottom: '12px' }}>Top Throwers</h2>
          <ConnectionTable rows={player.top_throwers} navigate={navigate} />
        </div>
      </div>

      {/* Synergy partners */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ marginBottom: '12px' }}>Best Synergy Partners</h2>
        <p style={{ color: '#888', fontSize: '12px', marginBottom: '10px' }}>All-time O-line synergy delta vs. individual scoring rates</p>
        {player.synergy_partners.length === 0 ? (
          <p style={{ color: '#888', fontSize: '14px' }}>No synergy data available.</p>
        ) : (
          <table style={{ borderCollapse: 'collapse', fontSize: '14px', minWidth: '400px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #444', color: '#aaa' }}>
                <th style={{ textAlign: 'left', paddingBottom: '8px', paddingRight: '16px' }}>Partner</th>
                <th style={{ textAlign: 'right', paddingBottom: '8px', paddingRight: '16px' }}>Delta</th>
                <th style={{ textAlign: 'right', paddingBottom: '8px', paddingRight: '16px' }}>Together</th>
                <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Shared Poss</th>
              </tr>
            </thead>
            <tbody>
              {player.synergy_partners.map((sp) => (
                <tr key={sp.id} style={{ borderBottom: '1px solid #2a2a3e' }}>
                  <td
                    style={{ padding: '8px 16px 8px 0', cursor: 'pointer', color: '#60a5fa' }}
                    onClick={() => navigate(`/player/${sp.id}`)}
                  >
                    {sp.name}
                  </td>
                  <td style={{ textAlign: 'right', padding: '8px 16px 8px 0', fontWeight: 600, color: synergyColor(sp.synergy_delta) }}>
                    {sp.synergy_delta > 0 ? '+' : ''}{(sp.synergy_delta * 100).toFixed(1)}pp
                  </td>
                  <td style={{ textAlign: 'right', padding: '8px 16px 8px 0', color: '#aaa' }}>
                    {pct(sp.combined_rate)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '8px 0', color: '#666' }}>
                    {sp.shared_possessions}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function ConnectionTable({ rows, navigate }: { rows: { id: string; name: string; count: number; completion_pct: number }[]; navigate: (path: string) => void }) {
  if (rows.length === 0) return <p style={{ color: '#888', fontSize: '14px' }}>No data.</p>
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
      <thead>
        <tr style={{ borderBottom: '1px solid #444', color: '#aaa' }}>
          <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Player</th>
          <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Count</th>
          <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Comp%</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} style={{ borderBottom: '1px solid #2a2a3e' }}>
            <td
              style={{ padding: '7px 0', cursor: 'pointer', color: '#60a5fa' }}
              onClick={() => navigate(`/player/${r.id}`)}
            >
              {r.name}
            </td>
            <td style={{ textAlign: 'right', padding: '7px 0', color: '#aaa' }}>{r.count}</td>
            <td style={{ textAlign: 'right', padding: '7px 0', fontWeight: 600 }}>{(r.completion_pct * 100).toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default PlayerPage

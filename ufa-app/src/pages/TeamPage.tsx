import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, TeamResponse, TeamGame } from '../api/client'
import { teamLabel } from '../utils'


function TeamPage() {
  const { teamId } = useParams<{ teamId: string }>()
  const navigate = useNavigate()
  const [team, setTeam] = useState<TeamResponse | null>(null)
  const [selectedYear, setSelectedYear] = useState<number | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const hasInitialized = useRef(false)

  useEffect(() => {
    if (!teamId) return
    let cancelled = false
    const fetchTeam = async () => {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getTeam(teamId, selectedYear)
        if (cancelled) return

        if (!hasInitialized.current && data.available_years.length > 0) {
          // First load: silently redirect to most recent year, keep spinner up
          hasInitialized.current = true
          setSelectedYear(data.available_years[data.available_years.length - 1])
        } else {
          setTeam(data)
          setLoading(false)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch team data')
          setLoading(false)
        }
      }
    }
    fetchTeam()
    return () => { cancelled = true }
  }, [teamId, selectedYear])

  const synergyColor = (delta: number) => {
    if (delta > 0.05) return '#22c55e'
    if (delta < -0.05) return '#ef4444'
    return '#f59e0b'
  }

  if (loading) return <div className="app"><p>Loading team data...</p></div>
  if (error) return <div className="app"><p style={{ color: 'red' }}>Error: {error}</p></div>
  if (!team) return null

  const { record } = team
  const winPct = record.games > 0 ? (record.wins / record.games * 100).toFixed(1) : '—'

  return (
    <div className="app">
      <button
        onClick={() => navigate('/')}
        style={{ marginBottom: '16px', padding: '6px 14px', cursor: 'pointer', borderRadius: '6px', border: '1px solid #555', backgroundColor: '#2a2a3e', color: 'white' }}
      >
        ← Back
      </button>

      <header style={{ marginBottom: '16px' }}>
        <h1>{team.team_name}</h1>
        <p style={{ color: '#aaa', fontSize: '14px' }}>{team.division} · {team.team_id}</p>
      </header>

      {/* Season selector */}
      {team.available_years.length > 0 && (
        <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '14px', color: '#aaa' }}>Season:</span>
          <button
            onClick={() => setSelectedYear(undefined)}
            style={{
              padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '13px',
              backgroundColor: selectedYear === undefined ? '#6366f1' : '#2a2a3e',
              color: 'white', border: '1px solid #555',
            }}
          >
            All
          </button>
          {team.available_years.map((y) => (
            <button
              key={y}
              onClick={() => setSelectedYear(y)}
              style={{
                padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '13px',
                backgroundColor: selectedYear === y ? '#6366f1' : '#2a2a3e',
                color: 'white', border: '1px solid #555',
              }}
            >
              {y}
            </button>
          ))}
        </div>
      )}

      {/* Record + O-line stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '28px' }}>
        {[
          { label: 'Wins', value: record.wins },
          { label: 'Losses', value: record.losses },
          { label: 'Win %', value: `${winPct}%` },
          { label: 'O-Line Hold Rate', value: `${(team.o_line_rate * 100).toFixed(1)}%` },
        ].map(({ label, value }) => (
          <div key={label} className="stat-card">
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        {/* Top Players */}
        <div>
          <h2 style={{ marginBottom: '12px' }}>Top Players by Hold Rate</h2>
          {team.top_players.length === 0 ? (
            <p style={{ color: '#888', fontSize: '14px' }}>No players meet the minimum possession threshold for this season.</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #444', color: '#aaa' }}>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Player</th>
                  <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Hold Rate</th>
                  <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Possessions</th>
                </tr>
              </thead>
              <tbody>
                {team.top_players.map((p) => (
                  <tr key={p.id} style={{ borderBottom: '1px solid #333' }}>
                    <td
                      style={{ padding: '8px 0', cursor: 'pointer', color: '#60a5fa' }}
                      onClick={() => navigate(`/player/${p.id}`)}
                    >{p.name}</td>
                    <td style={{ textAlign: 'right', padding: '8px 0', fontWeight: 600 }}>
                      {(p.hold_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right', padding: '8px 0', color: '#aaa' }}>
                      {p.possessions}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Top Synergies */}
        <div>
          <h2 style={{ marginBottom: '12px' }}>Top Pair Synergies</h2>
          {team.top_synergies.length === 0 ? (
            <p style={{ color: '#888', fontSize: '14px' }}>No pairs meet the minimum shared possession threshold for this season.</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #444', color: '#aaa' }}>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Pair</th>
                  <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Delta</th>
                  <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Together</th>
                  <th style={{ textAlign: 'right', paddingBottom: '8px' }}>Poss</th>
                </tr>
              </thead>
              <tbody>
                {team.top_synergies.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                    <td style={{ padding: '8px 0', lineHeight: '1.4' }}>
                      <span style={{ display: 'block' }}>{s.player1.name}</span>
                      <span style={{ display: 'block', color: '#aaa' }}>{s.player2.name}</span>
                    </td>
                    <td style={{ textAlign: 'right', padding: '8px 0', fontWeight: 600, color: synergyColor(s.synergy_delta) }}>
                      {s.synergy_delta > 0 ? '+' : ''}{(s.synergy_delta * 100).toFixed(1)}pp
                    </td>
                    <td style={{ textAlign: 'right', padding: '8px 0', color: '#aaa' }}>
                      {(s.combined_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right', padding: '8px 0', color: '#666' }}>
                      {s.shared_possessions}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Past Games */}
      {team.past_games.length > 0 && (() => {
        const gamesByYear: Record<number, TeamGame[]> = {}
        for (const g of team.past_games) {
          if (!gamesByYear[g.year]) gamesByYear[g.year] = []
          gamesByYear[g.year].push(g)
        }
        const years = Object.keys(gamesByYear).map(Number).sort((a, b) => b - a)
        return (
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ marginBottom: '12px' }}>Game History</h2>
            {years.map(yr => (
              <div key={yr} style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '15px', color: '#aaa', marginBottom: '8px' }}>{yr}</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #444', color: '#888' }}>
                      <th style={{ textAlign: 'left', paddingBottom: '6px' }}>Date</th>
                      <th style={{ textAlign: 'left', paddingBottom: '6px' }}>Opponent</th>
                      <th style={{ textAlign: 'right', paddingBottom: '6px' }}>Score</th>
                      <th style={{ textAlign: 'center', paddingBottom: '6px', width: '50px' }}>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gamesByYear[yr].map(g => {
                      const isHome = g.home_team_id === team.team_id
                      const opp = isHome ? g.away_team_id : g.home_team_id
                      const teamScore = isHome ? g.home_score : g.away_score
                      const oppScore = isHome ? g.away_score : g.home_score
                      return (
                        <tr key={g.game_id} style={{ borderBottom: '1px solid #2a2a3e', cursor: 'pointer' }}
                            onClick={() => navigate(`/game/${g.game_id}`)}>
                          <td style={{ padding: '6px 0', color: '#aaa' }}>{g.game_date}</td>
                          <td style={{ padding: '6px 0', color: '#60a5fa' }}
                              onClick={e => { e.stopPropagation(); navigate(`/team/${opp}`) }}>
                            {teamLabel(opp)}
                          </td>
                          <td style={{ textAlign: 'right', padding: '6px 0', fontVariantNumeric: 'tabular-nums' }}>
                            {teamScore}–{oppScore}
                          </td>
                          <td style={{ textAlign: 'center', padding: '6px 0' }}>
                            <span style={{
                              padding: '1px 7px', borderRadius: '4px', fontSize: '11px', fontWeight: 700,
                              backgroundColor: g.won ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                              color: g.won ? '#22c55e' : '#ef4444',
                            }}>{g.won ? 'W' : 'L'}</span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        )
      })()}

      {/* Roster */}
      <div>
        <h2 style={{ marginBottom: '12px' }}>
          Roster{selectedYear ? ` — ${selectedYear}` : ' — All-time'}
        </h2>
        {team.roster.length === 0 ? (
          <p style={{ color: '#888', fontSize: '14px' }}>No roster data for this season.</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #444', color: '#aaa' }}>
                <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Player</th>
                <th style={{ textAlign: 'center', paddingBottom: '8px', width: '60px' }}>Line</th>
                <th style={{ textAlign: 'right', paddingBottom: '8px' }}>O poss</th>
                <th style={{ textAlign: 'right', paddingBottom: '8px' }}>D poss</th>
              </tr>
            </thead>
            <tbody>
              {team.roster.map((p) => {
                const primaryLine = p.o_appearances >= p.d_appearances ? 'O' : 'D'
                return (
                  <tr key={p.id} style={{ borderBottom: '1px solid #2a2a3e' }}>
                    <td
                      style={{ padding: '7px 0', cursor: 'pointer', color: '#60a5fa' }}
                      onClick={() => navigate(`/player/${p.id}`)}
                    >
                      {p.name}
                    </td>
                    <td style={{ textAlign: 'center', padding: '7px 0' }}>
                      <span style={{
                        display: 'inline-block',
                        padding: '1px 7px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 700,
                        backgroundColor: primaryLine === 'O' ? 'rgba(99,102,241,0.2)' : 'rgba(239,68,68,0.2)',
                        color: primaryLine === 'O' ? '#818cf8' : '#f87171',
                      }}>
                        {primaryLine}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right', padding: '7px 0', color: '#aaa' }}>{p.o_appearances}</td>
                    <td style={{ textAlign: 'right', padding: '7px 0', color: '#aaa' }}>{p.d_appearances}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default TeamPage

import { useRef, useEffect, useCallback, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { PlayerStats } from '../api/client'

interface PlayerData {
  name: string
  x: number
  y: number
  cluster: number
}

interface PlayerEmbeddingsProps {
  players: string[]
  coordinates: number[][]
  clusters: number[]
  playerStats: Record<string, PlayerStats>
  nameMap: Record<string, string>
  selectedPlayer: string | null
  onPlayerClick?: (player: string) => void
}

const WIDTH = 900
const HEIGHT = 600
const MARGIN = { top: 20, right: 20, bottom: 20, left: 20 }
const INFO_PANEL_WIDTH = 200

// Distinct colors for clusters, -1 (noise) gets gray
const CLUSTER_COLORS = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
  '#469990', '#dcbeff', '#9a6324', '#800000', '#aaffc3',
  '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#e6beff',
]
const NOISE_COLOR = '#555'

// Archetypes derived from cluster mean stats (completion%, huck rate, avg throw dist)
const CLUSTER_LABELS: Record<number, string> = {
  0: 'Deep Thrower',   // 91.5% comp, 10.4% hucks, 18.4yd avg
  1: 'All-Around',     // 94.0% comp, 6.3% hucks, 16.3yd avg
  2: 'Handler',        // 93.8% comp, 5.8% hucks, 15.4yd avg
  3: 'Reset Handler',  // 96.0% comp, 2.8% hucks, 13.9yd avg
}

function getClusterColor(cluster: number): string {
  if (cluster < 0) return NOISE_COLOR
  return CLUSTER_COLORS[cluster % CLUSTER_COLORS.length]
}

function PlayerEmbeddings({ players, coordinates, clusters, playerStats, nameMap, selectedPlayer, onPlayerClick }: PlayerEmbeddingsProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [hoveredPlayer, setHoveredPlayer] = useState<string | null>(null)
  const [pinnedPlayer, setPinnedPlayer] = useState<string | null>(null)

  // The player shown in the side panel: pinned takes priority, then hovered
  const displayedPlayer = pinnedPlayer ?? hoveredPlayer

  // Memoize so D3 effect only re-runs when the actual data changes, not on every hover/pin state update
  const data: PlayerData[] = useMemo(() => players.map((name, i) => ({
    name,
    x: coordinates[i][0],
    y: coordinates[i][1],
    cluster: clusters[i],
  })), [players, coordinates, clusters])

  // Unique clusters for legend
const getNeighbors = useCallback(
    (playerName: string, count: number) => {
      const player = data.find((d) => d.name === playerName)
      if (!player) return []
      return data
        .filter((d) => d.name !== playerName)
        .map((d) => ({
          ...d,
          dist: Math.sqrt((d.x - player.x) ** 2 + (d.y - player.y) ** 2),
        }))
        .sort((a, b) => a.dist - b.dist)
        .slice(0, count)
    },
    [data]
  )

  // Initial render
  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Scales
    const xExtent = d3.extent(data, (d) => d.x) as [number, number]
    const yExtent = d3.extent(data, (d) => d.y) as [number, number]
    const xPad = (xExtent[1] - xExtent[0]) * 0.1
    const yPad = (yExtent[1] - yExtent[0]) * 0.1

    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - xPad, xExtent[1] + xPad])
      .range([MARGIN.left, WIDTH - MARGIN.right])

    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1] + yPad])
      .range([HEIGHT - MARGIN.bottom, MARGIN.top])

    // Container group for zoom transforms
    const g = svg.append('g').attr('class', 'zoom-group')

    // Circles colored by cluster
    g.selectAll('circle')
      .data(data)
      .join('circle')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', 5)
      .attr('fill', (d) => getClusterColor(d.cluster))
      .attr('fill-opacity', 0.8)
      .attr('stroke', 'none')
      .attr('cursor', 'pointer')
      .on('mouseenter', function (_, d) {
        const k = d3.zoomTransform(svgRef.current!).k
        d3.select(this).attr('r', 8 / k).attr('fill', 'white').attr('fill-opacity', 1)
        setHoveredPlayer(d.name)
      })
      .on('mouseleave', function (_, d) {
        const k = d3.zoomTransform(svgRef.current!).k
        const isSelected = d.name === selectedPlayer
        const isNeighbor =
          selectedPlayer && getNeighbors(selectedPlayer, 10).some((n) => n.name === d.name)
        d3.select(this)
          .attr('r', (isSelected ? 8 : isNeighbor ? 6 : 5) / k)
          .attr('fill', isSelected ? 'cyan' : getClusterColor(d.cluster))
          .attr('fill-opacity', isSelected ? 1 : isNeighbor ? 0.9 : 0.8)
        setHoveredPlayer(null)
      })
      .on('click', (event, d) => {
        event.stopPropagation()
        // Toggle pin: click same player to unpin, click different to pin
        setPinnedPlayer((prev) => (prev === d.name ? null : d.name))
        onPlayerClick?.(d.name)
      })

    // Labels (initially hidden)
    g.selectAll('.player-label')
      .data(data)
      .join('text')
      .attr('class', 'player-label')
      .attr('x', (d) => xScale(d.x) + 8)
      .attr('y', (d) => yScale(d.y) + 4)
      .text((d) => nameMap[d.name] ?? d.name)
      .attr('fill', 'white')
      .attr('font-size', '10px')
      .attr('font-family', 'monospace')
      .attr('pointer-events', 'none')
      .attr('opacity', 0)

    // Zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 20])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)

        const k = event.transform.k
        g.selectAll<SVGTextElement, PlayerData>('.player-label').attr('opacity', (d) => {
          if (d.name === selectedPlayer) return 1
          if (
            selectedPlayer &&
            getNeighbors(selectedPlayer, 10).some((n) => n.name === d.name)
          )
            return 1
          return k > 3 ? 0.8 : 0
        })

        // Scale circles inversely to maintain visual size
        const baseR = 5
        g.selectAll<SVGCircleElement, PlayerData>('circle').attr('r', (d) => {
          const isSelected = d.name === selectedPlayer
          const isNeighbor =
            selectedPlayer && getNeighbors(selectedPlayer, 10).some((n) => n.name === d.name)
          const r = isSelected ? 8 : isNeighbor ? 6 : baseR
          return r / k
        })

        g.selectAll<SVGTextElement, PlayerData>('.player-label')
          .attr('font-size', `${10 / k}px`)
          .attr('x', (d) => xScale(d.x) + 8 / k)
          .attr('y', (d) => yScale(d.y) + 4 / k)
      })

    svg.call(zoom as any)
    zoomRef.current = zoom

    ;(svgRef.current as any).__xScale = xScale
    ;(svgRef.current as any).__yScale = yScale
  }, [data, getNeighbors, onPlayerClick]) // eslint-disable-line react-hooks/exhaustive-deps

  // Zoom to selected player
  useEffect(() => {
    if (!selectedPlayer || !svgRef.current || !zoomRef.current) return

    const svg = d3.select(svgRef.current)
    const xScale = (svgRef.current as any).__xScale as d3.ScaleLinear<number, number>
    const yScale = (svgRef.current as any).__yScale as d3.ScaleLinear<number, number>
    if (!xScale || !yScale) return

    const player = data.find((d) => d.name === selectedPlayer)
    if (!player) return

    const neighbors = getNeighbors(selectedPlayer, 10)
    const px = xScale(player.x)
    const py = yScale(player.y)

    const zoomLevel = 5
    const transform = d3.zoomIdentity
      .translate(WIDTH / 2, HEIGHT / 2)
      .scale(zoomLevel)
      .translate(-px, -py)

    svg.transition().duration(750).call(zoomRef.current.transform, transform)

    const g = svg.select('.zoom-group')
    g.selectAll<SVGCircleElement, PlayerData>('circle')
      .attr('fill', (d) => {
        if (d.name === selectedPlayer) return 'cyan'
        return getClusterColor(d.cluster)
      })
      .attr('fill-opacity', (d) => {
        if (d.name === selectedPlayer) return 1
        if (neighbors.some((n) => n.name === d.name)) return 0.9
        return 0.8
      })
      .attr('stroke', (d) => (d.name === selectedPlayer ? 'white' : 'none'))
      .attr('stroke-width', (d) => (d.name === selectedPlayer ? 2 / zoomLevel : 0))

    g.selectAll<SVGTextElement, PlayerData>('.player-label').attr('opacity', (d) => {
      if (d.name === selectedPlayer) return 1
      if (neighbors.some((n) => n.name === d.name)) return 1
      return 0
    })
  }, [selectedPlayer, data, getNeighbors])

  return (
    <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
      {/* Graph */}
      <div style={{ position: 'relative', flex: '1 1 auto', minWidth: 0 }}>
        <svg
          ref={svgRef}
          width={WIDTH}
          height={HEIGHT}
          style={{
            backgroundColor: '#1a1a2e',
            borderRadius: '8px',
            maxWidth: '100%',
            height: 'auto',
            cursor: 'grab',
          }}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          onClick={() => setPinnedPlayer(null)}
        />

        {/* Cluster legend */}
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginTop: '10px' }}>
          {Object.entries(CLUSTER_LABELS).map(([c, label]) => (
            <div key={c} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: CLUSTER_COLORS[Number(c)], flexShrink: 0 }} />
              <span style={{ color: '#ccc', fontSize: '12px' }}>{label}</span>
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: NOISE_COLOR, flexShrink: 0 }} />
            <span style={{ color: '#ccc', fontSize: '12px' }}>Unclustered</span>
          </div>
        </div>
      </div>

      {/* Side panel — player stats */}
      <div
        style={{
          width: `${INFO_PANEL_WIDTH}px`,
          flexShrink: 0,
          backgroundColor: '#1a1a2e',
          borderRadius: '8px',
          padding: '12px 14px',
          fontFamily: 'monospace',
          minHeight: '200px',
          border: '1px solid #333',
        }}
      >
        {displayedPlayer ? (
          <>
            <div style={{ fontWeight: 'bold', color: 'white', fontSize: '14px', marginBottom: '4px' }}>
              {nameMap[displayedPlayer] ?? displayedPlayer}
              <span style={{ color: '#888', marginLeft: '8px', fontWeight: 'normal', fontSize: '12px' }}>
                {(() => {
                  const p = data.find((d) => d.name === displayedPlayer)
                  return p && p.cluster >= 0 ? (CLUSTER_LABELS[p.cluster] ?? `C${p.cluster}`) : 'unclustered'
                })()}
              </span>
            </div>
            {pinnedPlayer && (
              <div
                onClick={() => setPinnedPlayer(null)}
                style={{ color: '#888', fontSize: '10px', cursor: 'pointer', marginBottom: '6px' }}
              >
                click to unpin
              </div>
            )}
            {playerStats[displayedPlayer] && (
              <div style={{ color: '#ccc', lineHeight: '1.6', fontSize: '11px' }}>
                <div>{playerStats[displayedPlayer].total_throws} throws</div>
                <div>{playerStats[displayedPlayer].completion_pct}% completion</div>
                <div>{playerStats[displayedPlayer].avg_throw_dist}yd avg dist</div>
                <div>{playerStats[displayedPlayer].avg_throw_depth}yd avg depth</div>
                <div>{playerStats[displayedPlayer].huck_rate}% huck rate</div>
                <div>{playerStats[displayedPlayer].goal_pct}% goal rate</div>
                <div>{playerStats[displayedPlayer].avg_lateral_dist}yd lateral</div>
                <div>{playerStats[displayedPlayer].avg_dist_from_center}yd from center</div>
              </div>
            )}
          </>
        ) : (
          <div style={{ color: '#555', fontSize: '12px', textAlign: 'center', paddingTop: '80px' }}>
            Hover or click a player
          </div>
        )}
      </div>
    </div>
  )
}

export default PlayerEmbeddings

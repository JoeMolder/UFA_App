import { useState } from 'react'
import { ThrowTendencyBin } from '../api/client'

interface Props {
  bins: ThrowTendencyBin[]
  totalThrows: number
  maxAvgDist: number
  size?: number
}

const N_BINS = 16
const BIN_DEG = 360 / N_BINS
const BIN_RAD = (2 * Math.PI) / N_BINS

// Blue (#0ea5e9) → purple (#8b5cf6) based on frequency
function blueToP(t: number): string {
  const r = Math.round(14 + (139 - 14) * t)
  const g = Math.round(165 + (92 - 165) * t)
  const b = Math.round(233 + (246 - 233) * t)
  return `rgb(${r},${g},${b})`
}

function sectorPath(cx: number, cy: number, r: number, startA: number, endA: number): string {
  if (r < 1) return ''
  const x1 = cx + r * Math.cos(startA)
  const y1 = cy + r * Math.sin(startA)
  const x2 = cx + r * Math.cos(endA)
  const y2 = cy + r * Math.sin(endA)
  return `M ${cx} ${cy} L ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)} Z`
}

// Direction label for a bin angle (center of bin)
function dirLabel(angleDeg: number): string {
  const a = ((angleDeg % 360) + 360) % 360
  if (a < 22.5 || a >= 337.5) return 'Forward'
  if (a < 67.5) return 'Forward-right'
  if (a < 112.5) return 'Right'
  if (a < 157.5) return 'Back-right'
  if (a < 202.5) return 'Back'
  if (a < 247.5) return 'Back-left'
  if (a < 292.5) return 'Left'
  return 'Forward-left'
}

export function ThrowTendencyChart({ bins, totalThrows, maxAvgDist, size = 260 }: Props) {
  const [hovered, setHovered] = useState<number | null>(null)
  const pad = 28
  const cx = size / 2
  const cy = size / 2
  const maxR = size / 2 - pad

  const maxPct = Math.max(...bins.map((b) => b.pct), 0.0001)
  const hovBin = hovered !== null ? bins[hovered] : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
      <svg width={size} height={size} style={{ overflow: 'visible' }}>
        {/* Background disc */}
        <circle cx={cx} cy={cy} r={maxR} fill="rgba(255,255,255,0.025)" />

        {/* Grid rings at 25%, 50%, 75%, 100% */}
        {[0.25, 0.5, 0.75, 1.0].map((f) => (
          <circle key={f} cx={cx} cy={cy} r={maxR * f}
            fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
        ))}

        {/* Cardinal axis lines */}
        {[0, 90, 180, 270].map((deg) => {
          const a = (deg - 90) * Math.PI / 180
          return (
            <line key={deg}
              x1={cx} y1={cy}
              x2={cx + maxR * Math.cos(a)} y2={cy + maxR * Math.sin(a)}
              stroke="rgba(255,255,255,0.07)" strokeWidth={1}
            />
          )
        })}

        {/* Sectors */}
        {bins.map((bin, i) => {
          if (bin.count === 0) return null
          const r = (bin.avg_dist / maxAvgDist) * maxR
          const centerDeg = bin.angle_deg + BIN_DEG / 2
          const svgCenter = (centerDeg - 90) * Math.PI / 180
          const startA = svgCenter - BIN_RAD / 2
          const endA = svgCenter + BIN_RAD / 2
          const t = bin.pct / maxPct
          const isHov = hovered === i
          return (
            <path
              key={i}
              d={sectorPath(cx, cy, r, startA, endA)}
              fill={blueToP(t)}
              opacity={isHov ? 1 : 0.82}
              stroke={isHov ? 'rgba(255,255,255,0.4)' : 'none'}
              strokeWidth={isHov ? 1 : 0}
              style={{ cursor: 'pointer', transition: 'opacity 0.1s' }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            />
          )
        })}

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={3} fill="rgba(255,255,255,0.35)" />

        {/* Direction labels */}
        {[
          { label: '↑ fwd', deg: 0 },
          { label: 'right →', deg: 90 },
          { label: '← left', deg: 270 },
        ].map(({ label, deg }) => {
          const a = (deg - 90) * Math.PI / 180
          const lx = cx + (maxR + 14) * Math.cos(a)
          const ly = cy + (maxR + 14) * Math.sin(a)
          return (
            <text key={deg} x={lx} y={ly + 3}
              textAnchor="middle" fontSize={9} fill="rgba(255,255,255,0.3)"
              fontFamily="inherit"
            >
              {label}
            </text>
          )
        })}
      </svg>

      {/* Tooltip area */}
      <div style={{ height: '32px', fontSize: '12px', color: '#aaa', textAlign: 'center' }}>
        {hovBin && hovBin.count > 0 ? (
          <>
            <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{dirLabel(hovBin.angle_deg + BIN_DEG / 2)}</span>
            {' · '}
            {(hovBin.pct * 100).toFixed(1)}% of throws
            {' · '}
            avg {hovBin.avg_dist.toFixed(1)} yds
          </>
        ) : (
          <span style={{ color: '#444' }}>{totalThrows} completions · hover a sector</span>
        )}
      </div>

      {/* Color legend */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: '#555' }}>
        <span>rare</span>
        <svg width={80} height={8}>
          <defs>
            <linearGradient id="tendGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0ea5e9" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
          <rect x={0} y={0} width={80} height={8} rx={4} fill="url(#tendGrad)" />
        </svg>
        <span>frequent</span>
      </div>
    </div>
  )
}

import '../styles/ScoreTimeline.css'

interface Point {
  homeScore: number
  awayScore: number
  events: any[]
  startIndex: number
  endIndex: number
}

interface ScoreTimelineProps {
  points: Point[]
  selectedIndex: number
  onSelectPoint: (index: number) => void
  homeTeam: string
  awayTeam: string
}

function ScoreTimeline({ points, selectedIndex, onSelectPoint, homeTeam, awayTeam }: ScoreTimelineProps) {
  return (
    <div className="score-timeline">
      <h3>Point Timeline</h3>
      <div className="timeline-container">
        {points.map((point, index) => {
          const isSelected = index === selectedIndex
          const scoreLabel = `${point.awayScore}-${point.homeScore}`

          return (
            <div
              key={index}
              className={`timeline-point ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectPoint(index)}
              title={`${awayTeam} ${point.awayScore} - ${homeTeam} ${point.homeScore}`}
            >
              <div className="point-score">{scoreLabel}</div>
              <div className="point-number">Point {index + 1}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ScoreTimeline

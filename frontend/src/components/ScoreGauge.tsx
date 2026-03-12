interface ScoreGaugeProps {
  score: number  // 0–1
  label: string
  size?: number
}

export default function ScoreGauge({ score, label, size = 100 }: ScoreGaugeProps) {
  const radius = (size - 10) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - Math.max(0, Math.min(1, score)))

  const color = score >= 0.8 ? '#22c55e' : score >= 0.5 ? '#eab308' : '#ef4444'
  const bgColor = score >= 0.8 ? '#dcfce7' : score >= 0.5 ? '#fef9c3' : '#fee2e2'

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth={8}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={8}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div
          className="absolute inset-0 flex items-center justify-center rounded-full"
          style={{ backgroundColor: bgColor }}
        >
          <span className="text-lg font-bold" style={{ color }}>
            {Math.round(score * 100)}%
          </span>
        </div>
      </div>
      <span className="text-xs font-medium text-gray-600">{label}</span>
    </div>
  )
}

import { Shield, ShieldAlert, ShieldCheck, ShieldX } from 'lucide-react'
import ScoreGauge from './ScoreGauge'

interface ModalityScore {
  modality: string
  score: number
  passed: boolean
}

interface VerificationResultProps {
  status: 'verified' | 'rejected' | 'suspicious'
  overallScore: number
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
  scores: ModalityScore[]
  details?: Record<string, unknown>
}

const riskColors = {
  low: 'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-orange-100 text-orange-800',
  critical: 'bg-red-100 text-red-800',
}

const statusConfig = {
  verified: { icon: ShieldCheck, color: 'text-green-600', bg: 'bg-green-50 border-green-200', label: 'Verified' },
  rejected: { icon: ShieldX, color: 'text-red-600', bg: 'bg-red-50 border-red-200', label: 'Rejected' },
  suspicious: { icon: ShieldAlert, color: 'text-yellow-600', bg: 'bg-yellow-50 border-yellow-200', label: 'Suspicious' },
}

export default function VerificationResult({ status, overallScore, riskLevel, scores, details }: VerificationResultProps) {
  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <div className={`border rounded-2xl p-6 ${config.bg} space-y-6`}>
      {/* Header */}
      <div className="flex items-center gap-4">
        <Icon className={`h-12 w-12 ${config.color}`} />
        <div>
          <h3 className={`text-xl font-bold ${config.color}`}>{config.label}</h3>
          <span className={`inline-block mt-1 px-3 py-0.5 rounded-full text-xs font-semibold ${riskColors[riskLevel]}`}>
            Risk: {riskLevel.toUpperCase()}
          </span>
        </div>
        <div className="ml-auto">
          <ScoreGauge score={overallScore} label="Overall" size={90} />
        </div>
      </div>

      {/* Modality scores */}
      {scores.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Modality Scores</h4>
          <div className="flex flex-wrap gap-4 justify-center">
            {scores.map((s) => (
              <div key={s.modality} className="flex flex-col items-center">
                <ScoreGauge score={s.score} label={s.modality} size={72} />
                <span className={`mt-1 text-xs font-medium ${s.passed ? 'text-green-600' : 'text-red-600'}`}>
                  {s.passed ? 'PASS' : 'FAIL'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Details section */}
      {details && Object.keys(details).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Details</h4>
          <div className="bg-white/60 rounded-lg p-3 text-xs font-mono overflow-auto max-h-40">
            <pre>{JSON.stringify(details, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

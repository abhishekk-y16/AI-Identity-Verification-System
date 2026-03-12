import { useState, useEffect } from 'react'
import { attendanceAPI } from '../services/api'
import { CalendarDays, Clock, TrendingUp } from 'lucide-react'

export default function MyAttendancePage() {
  const [records, setRecords] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [startDate, setStartDate] = useState(() => {
    const d = new Date()
    d.setDate(1)
    return d.toISOString().split('T')[0]
  })
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0])

  useEffect(() => {
    setLoading(true)
    attendanceAPI.getHistory(startDate, endDate)
      .then(res => setRecords(res.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [startDate, endDate])

  // Group records by date
  const grouped = records.reduce((acc: Record<string, any[]>, r: any) => {
    const d = new Date(r.timestamp).toLocaleDateString()
    if (!acc[d]) acc[d] = []
    acc[d].push(r)
    return acc
  }, {})

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">My Attendance</h1>
        <p className="text-gray-500 mt-1">View your attendance history</p>
      </div>

      {/* Date filters */}
      <div className="flex gap-4 flex-wrap">
        <div>
          <label className="block text-sm text-gray-600 mb-1">From</label>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm" />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">To</label>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm" />
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full" /></div>
      ) : Object.keys(grouped).length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm border p-8 text-center text-gray-500">
          No attendance records found for this period.
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(grouped).map(([date, recs]) => (
            <div key={date} className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <CalendarDays className="h-4 w-4 text-blue-600" />
                {date}
              </h3>
              <div className="space-y-2">
                {(recs as any[]).map((r: any) => (
                  <div key={r.record_id} className="flex items-center justify-between py-2 border-b last:border-0">
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        r.punch_type === 'clock_in' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                      }`}>
                        {r.punch_type === 'clock_in' ? 'IN' : 'OUT'}
                      </span>
                      <span className="text-sm">{new Date(r.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">
                        Face: {r.face_score ? (r.face_score * 100).toFixed(0) + '%' : '-'}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        r.status === 'on_time' ? 'bg-green-50 text-green-600' :
                        r.status === 'late' ? 'bg-yellow-50 text-yellow-600' :
                        r.status === 'early_departure' ? 'bg-orange-50 text-orange-600' :
                        'bg-gray-50 text-gray-600'
                      }`}>
                        {r.status?.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

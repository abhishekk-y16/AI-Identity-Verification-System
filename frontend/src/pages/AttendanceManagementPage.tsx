import { useState, useEffect } from 'react'
import { attendanceAPI } from '../services/api'
import { RefreshCw } from 'lucide-react'

export default function AttendanceManagementPage() {
  const [records, setRecords] = useState<any[]>([])
  const [date, setDate] = useState(new Date().toISOString().split('T')[0])
  const [loading, setLoading] = useState(true)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [message, setMessage] = useState('')

  const fetchRecords = () => {
    setLoading(true)
    attendanceAPI.getHistory(date, date)
      .then(r => setRecords(Array.isArray(r.data) ? r.data : []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchRecords() }, [date])

  const triggerSummary = async () => {
    setSummaryLoading(true)
    setMessage('')
    try {
      await attendanceAPI.computeDailySummary(date)
      setMessage('Daily summaries computed successfully')
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to compute summaries')
    } finally {
      setSummaryLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Attendance Management</h1>
          <p className="text-gray-500 mt-1">View and Manage employee attendance</p>
        </div>
        <button onClick={triggerSummary} disabled={summaryLoading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          <RefreshCw className={`h-4 w-4 ${summaryLoading ? 'animate-spin' : ''}`} />
          Compute Daily Summary
        </button>
      </div>

      {message && <div className="p-3 bg-blue-50 text-blue-700 rounded-lg text-sm">{message}</div>}

      <div className="flex items-center gap-3">
        <input type="date" value={date} onChange={e => setDate(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm" />
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-12"><div className="animate-spin h-6 w-6 border-b-2 border-blue-600 rounded-full" /></div>
        ) : records.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">No Records for this Date</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Face Score</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Voice Score</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">IP</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {records.map((r: any) => (
                  <tr key={r.record_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium text-gray-900">{r.user_id}</td>
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${r.punch_type === 'clock_in' ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'}`}>
                        {r.punch_type === 'clock_in' ? 'Clock In' : 'Clock Out'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-500">{new Date(r.timestamp).toLocaleTimeString()}</td>
                    <td className="px-4 py-3 text-gray-500">{r.face_score?.toFixed(2) || '—'}</td>
                    <td className="px-4 py-3 text-gray-500">{r.voice_score?.toFixed(2) || '—'}</td>
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        r.status === 'on_time' ? 'bg-green-100 text-green-700' :
                        r.status === 'late' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-gray-100 text-gray-500'
                      }`}>{r.status}</span>
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">{r.ip_address || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

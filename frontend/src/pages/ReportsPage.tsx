import { useState } from 'react'
import { reportAPI } from '../services/api'
import { Download, Search } from 'lucide-react'

export default function ReportsPage() {
  const [startDate, setStartDate] = useState(() => {
    const d = new Date(); d.setDate(1); return d.toISOString().split('T')[0]
  })
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0])
  const [report, setReport] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)

  const fetchReport = () => {
    setLoading(true)
    reportAPI.attendance(startDate, endDate)
      .then(r => setReport(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  const exportCSV = async () => {
    setExporting(true)
    try {
      const response = await reportAPI.exportCSV(startDate, endDate)
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `attendance_${startDate}_${endDate}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      alert('Export failed')
    } finally {
      setExporting(false)
    }
  }

  const summaries = report?.summaries || report?.data || (Array.isArray(report) ? report : [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Reports</h1>
        <p className="text-gray-500 mt-1">Generate attendance reports and export data</p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Start Date</label>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm" />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">End Date</label>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm" />
        </div>
        <button onClick={fetchReport} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          <Search className="h-4 w-4" /> Generate
        </button>
        <button onClick={exportCSV} disabled={exporting}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50">
          <Download className="h-4 w-4" /> {exporting ? 'Exporting...' : 'Export CSV'}
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><div className="animate-spin h-6 w-6 border-b-2 border-blue-600 rounded-full" /></div>
      ) : summaries.length > 0 ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Days Present</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Days Absent</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Late Days</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Hours</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Overtime Hours</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Avg Hours/Day</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {summaries.map((s: any, i: number) => (
                  <tr key={s.user_id || i} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium text-gray-900">{s.employee_name || s.user_id}</td>
                    <td className="px-4 py-3 text-green-600 font-medium">{s.days_present ?? s.present_days ?? 0}</td>
                    <td className="px-4 py-3 text-red-600 font-medium">{s.days_absent ?? s.absent_days ?? 0}</td>
                    <td className="px-4 py-3 text-yellow-600 font-medium">{s.late_days ?? 0}</td>
                    <td className="px-4 py-3 text-gray-700">{(s.total_hours ?? 0).toFixed(1)}</td>
                    <td className="px-4 py-3 text-blue-600">{(s.overtime_hours ?? 0).toFixed(1)}</td>
                    <td className="px-4 py-3 text-gray-500">{(s.average_hours ?? 0).toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : report !== null ? (
        <div className="text-center py-12 text-gray-500 text-sm">No data for selected period</div>
      ) : null}
    </div>
  )
}

import { useState, useEffect } from 'react'
import { officeAPI } from '../services/api'
import { Users, UserCheck, UserX, Clock, AlertTriangle, TrendingUp } from 'lucide-react'

export default function AdminDashboardPage() {
  const [stats, setStats] = useState<any>(null)
  const [departments, setDepartments] = useState<any[]>([])
  const [timeseries, setTimeseries] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      officeAPI.stats(),
      officeAPI.departmentBreakdown(),
      officeAPI.timeseries(7),
    ]).then(([s, d, t]) => {
      setStats(s.data)
      setDepartments(d.data)
      setTimeseries(t.data)
    }).catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="flex justify-center py-12"><div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full" /></div>

  const cards = stats ? [
    { label: 'Total Employees', value: stats.total_employees, icon: Users, color: 'bg-blue-50 text-blue-600' },
    { label: 'Present Today', value: stats.present_today, icon: UserCheck, color: 'bg-green-50 text-green-600' },
    { label: 'Absent Today', value: stats.absent_today, icon: UserX, color: 'bg-red-50 text-red-600' },
    { label: 'Late Today', value: stats.late_today, icon: Clock, color: 'bg-yellow-50 text-yellow-600' },
    { label: 'On Leave', value: stats.on_leave_today, icon: AlertTriangle, color: 'bg-purple-50 text-purple-600' },
    { label: 'Avg Hours', value: stats.average_hours_today?.toFixed(1) || '0', icon: TrendingUp, color: 'bg-indigo-50 text-indigo-600' },
  ] : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">Office attendance overview</p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {cards.map(c => (
          <div key={c.label} className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <div className={`inline-flex p-2 rounded-lg ${c.color} mb-2`}>
              <c.icon className="h-5 w-5" />
            </div>
            <p className="text-2xl font-bold text-gray-900">{c.value}</p>
            <p className="text-xs text-gray-500 mt-0.5">{c.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Department Breakdown */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">Department Breakdown</h3>
          </div>
          {departments.length === 0 ? (
            <div className="p-6 text-center text-gray-500 text-sm">No departments yet</div>
          ) : (
            <div className="divide-y divide-gray-100">
              {departments.map((d: any) => (
                <div key={d.department_id || d.name} className="px-4 py-3 flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900">{d.name}</p>
                  <div className="flex gap-3 text-xs">
                    <span className="text-green-600 font-medium">{d.present || 0} present</span>
                    <span className="text-red-600 font-medium">{d.absent || 0} absent</span>
                    <span className="text-yellow-600 font-medium">{d.late || 0} late</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 7 Day Trend */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">7-Day Attendance Trend</h3>
          </div>
          {timeseries.length === 0 ? (
            <div className="p-6 text-center text-gray-500 text-sm">No data available</div>
          ) : (
            <div className="p-4">
              <div className="space-y-2">
                {timeseries.map((pt: any) => {
                  const max = Math.max(...timeseries.map((t: any) => t.present || t.count || 1))
                  const val = pt.present || pt.count || 0
                  return (
                    <div key={pt.date} className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-20 shrink-0">{new Date(pt.date).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}</span>
                      <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden">
                        <div className="h-full bg-blue-500 rounded transition-all" style={{ width: `${(val / max) * 100}%` }} />
                      </div>
                      <span className="text-xs font-medium text-gray-700 w-8 text-right">{val}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

import { useState, useEffect } from 'react'
import { alertAPI } from '../services/api'
import { Bell, CheckCheck, AlertTriangle, Info, AlertCircle } from 'lucide-react'

const severityConfig: Record<string, { icon: any; color: string }> = {
  critical: { icon: AlertCircle, color: 'bg-red-50 border-red-200 text-red-700' },
  warning: { icon: AlertTriangle, color: 'bg-yellow-50 border-yellow-200 text-yellow-700' },
  info: { icon: Info, color: 'bg-blue-50 border-blue-200 text-blue-700' },
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 20

  const fetchAlerts = () => {
    setLoading(true)
    alertAPI.list({ page, page_size: pageSize })
      .then(r => {
        const data = r.data
        if (data.items) { setAlerts(data.items); setTotal(data.total || data.items.length) }
        else if (Array.isArray(data)) { setAlerts(data); setTotal(data.length) }
        else { setAlerts([]); setTotal(0) }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchAlerts() }, [page])

  const markRead = async (id: string) => {
    try { await alertAPI.markRead(id); fetchAlerts() } catch {}
  }

  const markAllRead = async () => {
    try { await alertAPI.markAllRead(); fetchAlerts() } catch {}
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
          <p className="text-gray-500 mt-1">System alerts and notifications</p>
        </div>
        <button onClick={markAllRead} className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200">
          <CheckCheck className="h-4 w-4" /> Mark All Read
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        {loading ? (
          <div className="flex justify-center py-12"><div className="animate-spin h-6 w-6 border-b-2 border-blue-600 rounded-full" /></div>
        ) : alerts.length === 0 ? (
          <div className="p-8 text-center">
            <Bell className="h-10 w-10 text-gray-300 mx-auto mb-2" />
            <p className="text-gray-500 text-sm">No alerts</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {alerts.map((a: any) => {
              const cfg = severityConfig[a.severity] || severityConfig.info
              const Icon = cfg.icon
              return (
                <div key={a.alert_id} className={`px-4 py-3 flex gap-3 ${!a.is_read ? 'bg-blue-50/30' : ''}`}>
                  <div className={`shrink-0 p-1.5 rounded-lg border ${cfg.color}`}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-gray-900">{a.title}</p>
                      {!a.is_read && <span className="h-2 w-2 bg-blue-500 rounded-full shrink-0" />}
                    </div>
                    <p className="text-sm text-gray-600 mt-0.5">{a.message}</p>
                    <p className="text-xs text-gray-400 mt-1">{new Date(a.created_at).toLocaleString()}</p>
                  </div>
                  {!a.is_read && (
                    <button onClick={() => markRead(a.alert_id)} className="text-xs text-blue-600 hover:underline shrink-0 self-start">
                      Mark read
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {total > pageSize && (
          <div className="px-4 py-3 border-t border-gray-200 flex items-center justify-between text-sm">
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
              className="px-3 py-1 rounded bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50">Previous</button>
            <span className="text-gray-500">Page {page}</span>
            <button onClick={() => setPage(p => p + 1)} disabled={page * pageSize >= total}
              className="px-3 py-1 rounded bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50">Next</button>
          </div>
        )}
      </div>
    </div>
  )
}

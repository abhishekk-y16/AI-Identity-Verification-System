import { useState, useEffect } from 'react'
import { leaveAPI } from '../services/api'
import { CheckCircle, XCircle, Clock } from 'lucide-react'

export default function LeaveManagementPage() {
  const [requests, setRequests] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)

  const fetchPending = () => {
    setLoading(true)
    leaveAPI.getPending()
      .then(r => setRequests(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchPending() }, [])

  const handleAction = async (id: string, action: 'approve' | 'reject') => {
    const remarks = action === 'reject' ? prompt('Reason for rejection:') : ''
    if (action === 'reject' && remarks === null) return
    setActionLoading(id)
    try {
      if (action === 'approve') await leaveAPI.approve(id, remarks || '')
      else await leaveAPI.reject(id, remarks || '')
      fetchPending()
    } catch (err: any) {
      alert(err.response?.data?.detail || `Failed to ${action}`)
    } finally {
      setActionLoading(null)
    }
  }

  if (loading) return <div className="flex justify-center py-12"><div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full" /></div>

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Leave Management</h1>
        <p className="text-gray-500 mt-1">Review and approve leave requests</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
          <h3 className="font-semibold text-gray-900">Pending Requests</h3>
          <span className="text-xs text-gray-500">{requests.length} pending</span>
        </div>
        {requests.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">No pending leave requests</div>
        ) : (
          <div className="divide-y divide-gray-100">
            {requests.map((r: any) => (
              <div key={r.request_id} className="px-4 py-4 flex flex-col sm:flex-row sm:items-center gap-3">
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{r.user_id}</p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {new Date(r.start_date).toLocaleDateString()} — {new Date(r.end_date).toLocaleDateString()} · {r.days_count} day(s)
                  </p>
                  <p className="text-sm text-gray-600 mt-1">{r.reason}</p>
                </div>
                <div className="flex gap-2 shrink-0">
                  <button onClick={() => handleAction(r.request_id, 'approve')}
                    disabled={actionLoading === r.request_id}
                    className="flex items-center gap-1 px-3 py-1.5 bg-green-600 text-white rounded-lg text-xs font-medium hover:bg-green-700 disabled:opacity-50">
                    <CheckCircle className="h-3.5 w-3.5" /> Approve
                  </button>
                  <button onClick={() => handleAction(r.request_id, 'reject')}
                    disabled={actionLoading === r.request_id}
                    className="flex items-center gap-1 px-3 py-1.5 bg-red-600 text-white rounded-lg text-xs font-medium hover:bg-red-700 disabled:opacity-50">
                    <XCircle className="h-3.5 w-3.5" /> Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

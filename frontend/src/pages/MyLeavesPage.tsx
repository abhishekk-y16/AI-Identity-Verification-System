import { useState, useEffect } from 'react'
import { leaveAPI } from '../services/api'
import { Timer, Plus, CheckCircle, XCircle, Clock } from 'lucide-react'

export default function MyLeavesPage() {
  const [balances, setBalances] = useState<any[]>([])
  const [requests, setRequests] = useState<any[]>([])
  const [leaveTypes, setLeaveTypes] = useState<any[]>([])
  const [showForm, setShowForm] = useState(false)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const [form, setForm] = useState({
    leave_type_id: '',
    start_date: '',
    end_date: '',
    reason: '',
  })

  useEffect(() => {
    Promise.all([
      leaveAPI.getMyBalance(),
      leaveAPI.getMyRequests(),
      leaveAPI.getTypes(),
    ]).then(([bal, req, types]) => {
      setBalances(bal.data)
      setRequests(req.data)
      setLeaveTypes(types.data)
    }).catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      await leaveAPI.requestLeave(form)
      const [req, bal] = await Promise.all([leaveAPI.getMyRequests(), leaveAPI.getMyBalance()])
      setRequests(req.data)
      setBalances(bal.data)
      setShowForm(false)
      setForm({ leave_type_id: '', start_date: '', end_date: '', reason: '' })
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit request')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) return <div className="flex justify-center py-12"><div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full" /></div>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Leaves</h1>
          <p className="text-gray-500 mt-1">Manage your leave balance and requests</p>
        </div>
        <button onClick={() => setShowForm(!showForm)} className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700">
          <Plus className="h-4 w-4" /> Apply Leave
        </button>
      </div>

      {/* Balances */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {balances.map((b: any) => (
          <div key={b.balance_id} className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <p className="text-sm text-gray-500">{leaveTypes.find((t: any) => t.leave_type_id === b.leave_type_id)?.name || 'Leave'}</p>
            <div className="flex items-end gap-1 mt-1">
              <span className="text-2xl font-bold text-gray-900">{b.remaining_days}</span>
              <span className="text-sm text-gray-400 mb-1">/ {b.total_days} days</span>
            </div>
            <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${(b.remaining_days / b.total_days) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>

      {/* Apply form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
          <h3 className="font-semibold text-gray-900">New Leave Request</h3>
          {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}
          <div>
            <label className="block text-sm text-gray-600 mb-1">Leave Type</label>
            <select value={form.leave_type_id} onChange={e => setForm({ ...form, leave_type_id: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required>
              <option value="">Select type...</option>
              {leaveTypes.map((t: any) => (
                <option key={t.leave_type_id} value={t.leave_type_id}>{t.name}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Start Date</label>
              <input type="date" value={form.start_date} onChange={e => setForm({ ...form, start_date: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">End Date</label>
              <input type="date" value={form.end_date} onChange={e => setForm({ ...form, end_date: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">Reason</label>
            <textarea value={form.reason} onChange={e => setForm({ ...form, reason: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" rows={3} required />
          </div>
          <div className="flex gap-3">
            <button type="submit" disabled={submitting} className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
              {submitting ? 'Submitting...' : 'Submit Request'}
            </button>
            <button type="button" onClick={() => setShowForm(false)} className="px-6 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm hover:bg-gray-200">
              Cancel
            </button>
          </div>
        </form>
      )}

      {/* Requests history */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200">
          <h3 className="font-semibold text-gray-900">Leave Requests</h3>
        </div>
        {requests.length === 0 ? (
          <div className="p-6 text-center text-gray-500 text-sm">No leave requests yet</div>
        ) : (
          <div className="divide-y divide-gray-100">
            {requests.map((r: any) => (
              <div key={r.request_id} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{r.reason}</p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {new Date(r.start_date).toLocaleDateString()} — {new Date(r.end_date).toLocaleDateString()} · {r.days_count} day(s)
                  </p>
                </div>
                <span className={`flex items-center gap-1 text-xs px-2.5 py-1 rounded-full font-medium ${
                  r.status === 'approved' ? 'bg-green-100 text-green-700' :
                  r.status === 'rejected' ? 'bg-red-100 text-red-700' :
                  'bg-yellow-100 text-yellow-700'
                }`}>
                  {r.status === 'approved' ? <CheckCircle className="h-3 w-3" /> :
                   r.status === 'rejected' ? <XCircle className="h-3 w-3" /> :
                   <Clock className="h-3 w-3" />}
                  {r.status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

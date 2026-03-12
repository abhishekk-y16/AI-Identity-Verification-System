import { useState, useEffect } from 'react'
import { shiftAPI } from '../services/api'
import { Plus, Pencil, Trash2 } from 'lucide-react'

export default function ShiftManagementPage() {
  const [shifts, setShifts] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editId, setEditId] = useState<string | null>(null)
  const [form, setForm] = useState({ name: '', start_time: '09:00', end_time: '18:00', grace_minutes: 15 })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const fetchShifts = () => {
    setLoading(true)
    shiftAPI.list()
      .then(r => setShifts(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchShifts() }, [])

  const openCreate = () => { setEditId(null); setForm({ name: '', start_time: '09:00', end_time: '18:00', grace_minutes: 15 }); setShowForm(true); setError('') }
  const openEdit = (s: any) => { setEditId(s.shift_id); setForm({ name: s.name, start_time: s.start_time, end_time: s.end_time, grace_minutes: s.grace_minutes }); setShowForm(true); setError('') }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true); setError('')
    try {
      if (editId) await shiftAPI.update(editId, form)
      else await shiftAPI.create(form)
      setShowForm(false)
      fetchShifts()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save')
    } finally { setSaving(false) }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this shift?')) return
    try {
      await shiftAPI.remove(id)
      fetchShifts()
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Cannot delete')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Shifts</h1>
          <p className="text-gray-500 mt-1">Manage work shifts and schedules</p>
        </div>
        <button onClick={openCreate} className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700">
          <Plus className="h-4 w-4" /> Add Shift
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
          <h3 className="font-semibold text-gray-900">{editId ? 'Edit' : 'New'} Shift</h3>
          {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}
          <div>
            <label className="block text-sm text-gray-600 mb-1">Name</label>
            <input type="text" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Start Time</label>
              <input type="time" value={form.start_time} onChange={e => setForm({ ...form, start_time: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">End Time</label>
              <input type="time" value={form.end_time} onChange={e => setForm({ ...form, end_time: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Grace (min)</label>
              <input type="number" value={form.grace_minutes} onChange={e => setForm({ ...form, grace_minutes: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" min={0} />
            </div>
          </div>
          <div className="flex gap-3">
            <button type="submit" disabled={saving} className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
              {saving ? 'Saving...' : 'Save'}
            </button>
            <button type="button" onClick={() => setShowForm(false)} className="px-6 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm hover:bg-gray-200">Cancel</button>
          </div>
        </form>
      )}

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-12"><div className="animate-spin h-6 w-6 border-b-2 border-blue-600 rounded-full" /></div>
        ) : shifts.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">No shifts configured yet</div>
        ) : (
          <div className="divide-y divide-gray-100">
            {shifts.map((s: any) => (
              <div key={s.shift_id} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{s.name}</p>
                  <p className="text-xs text-gray-500 mt-0.5">{s.start_time} — {s.end_time} · Grace: {s.grace_minutes}min</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${s.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                    {s.is_active ? 'Active' : 'Inactive'}
                  </span>
                  <button onClick={() => openEdit(s)} className="p-1.5 text-gray-400 hover:text-blue-600 rounded">
                    <Pencil className="h-4 w-4" />
                  </button>
                  <button onClick={() => handleDelete(s.shift_id)} className="p-1.5 text-gray-400 hover:text-red-600 rounded">
                    <Trash2 className="h-4 w-4" />
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

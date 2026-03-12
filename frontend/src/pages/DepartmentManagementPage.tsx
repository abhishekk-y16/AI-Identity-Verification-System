import { useState, useEffect } from 'react'
import { departmentAPI } from '../services/api'
import { Plus, Pencil, Trash2 } from 'lucide-react'

export default function DepartmentManagementPage() {
  const [departments, setDepartments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editId, setEditId] = useState<string | null>(null)
  const [form, setForm] = useState({ name: '', description: '' })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const fetch = () => {
    setLoading(true)
    departmentAPI.list()
      .then(r => setDepartments(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetch() }, [])

  const openCreate = () => { setEditId(null); setForm({ name: '', description: '' }); setShowForm(true); setError('') }
  const openEdit = (d: any) => { setEditId(d.department_id); setForm({ name: d.name, description: d.description || '' }); setShowForm(true); setError('') }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true); setError('')
    try {
      if (editId) await departmentAPI.update(editId, form)
      else await departmentAPI.create(form)
      setShowForm(false)
      fetch()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save')
    } finally { setSaving(false) }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this department?')) return
    try {
      await departmentAPI.remove(id)
      fetch()
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Cannot delete')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Departments</h1>
          <p className="text-gray-500 mt-1">Manage office departments</p>
        </div>
        <button onClick={openCreate} className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700">
          <Plus className="h-4 w-4" /> Add Department
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
          <h3 className="font-semibold text-gray-900">{editId ? 'Edit' : 'New'} Department</h3>
          {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}
          <div>
            <label className="block text-sm text-gray-600 mb-1">Name</label>
            <input type="text" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" required />
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">Description</label>
            <textarea value={form.description} onChange={e => setForm({ ...form, description: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" rows={2} />
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
        ) : departments.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">No departments yet</div>
        ) : (
          <div className="divide-y divide-gray-100">
            {departments.map((d: any) => (
              <div key={d.department_id} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{d.name}</p>
                  {d.description && <p className="text-xs text-gray-500 mt-0.5">{d.description}</p>}
                </div>
                <div className="flex gap-2">
                  <button onClick={() => openEdit(d)} className="p-1.5 text-gray-400 hover:text-blue-600 rounded">
                    <Pencil className="h-4 w-4" />
                  </button>
                  <button onClick={() => handleDelete(d.department_id)} className="p-1.5 text-gray-400 hover:text-red-600 rounded">
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

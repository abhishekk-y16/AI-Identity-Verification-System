import { useState, useEffect } from 'react'
import { employeeAPI, departmentAPI } from '../services/api'
import { Search } from 'lucide-react'

export default function EmployeeDirectoryPage() {
  const [employees, setEmployees] = useState<any[]>([])
  const [departments, setDepartments] = useState<any[]>([])
  const [search, setSearch] = useState('')
  const [deptFilter, setDeptFilter] = useState('')
  const [loading, setLoading] = useState(true)

  const fetchEmployees = () => {
    setLoading(true)
    const params: any = {}
    if (search) params.search = search
    if (deptFilter) params.department_id = deptFilter
    employeeAPI.directory(params.department_id, params.search)
      .then(r => setEmployees(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    departmentAPI.list().then(r => setDepartments(r.data)).catch(() => {})
  }, [])

  useEffect(() => {
    const timer = setTimeout(fetchEmployees, 300)
    return () => clearTimeout(timer)
  }, [search, deptFilter])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Employee Directory</h1>
        <p className="text-gray-500 mt-1">View and manage employee profiles</p>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input type="text" value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search by name, email, or code..."
            className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg text-sm" />
        </div>
        <select value={deptFilter} onChange={e => setDeptFilter(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm">
          <option value="">All Departments</option>
          {departments.map((d: any) => (
            <option key={d.department_id} value={d.department_id}>{d.name}</option>
          ))}
        </select>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-12"><div className="animate-spin h-6 w-6 border-b-2 border-blue-600 rounded-full" /></div>
        ) : employees.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">No employees found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Code</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Department</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Designation</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {employees.map((emp: any) => (
                  <tr key={emp.profile_id || emp.user_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium text-gray-900">{emp.full_name || emp.name}</td>
                    <td className="px-4 py-3 text-gray-500">{emp.employee_code}</td>
                    <td className="px-4 py-3 text-gray-500">{emp.email}</td>
                    <td className="px-4 py-3 text-gray-500">{emp.department_name || '—'}</td>
                    <td className="px-4 py-3 text-gray-500">{emp.designation || '—'}</td>
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${emp.is_active !== false ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                        {emp.is_active !== false ? 'Active' : 'Inactive'}
                      </span>
                    </td>
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

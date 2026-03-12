import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT token to every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ─── Auth ──────────────────────────────────────────────────────
export const authAPI = {
  register: (data: { name: string; email: string; password: string }) =>
    client.post('/auth/register', data),
  login: (data: { email: string; password: string }) =>
    client.post('/auth/login', data),
  getMe: () => client.get('/auth/me'),
}

// ─── Face (biometric enrollment) ───────────────────────────────
export const faceAPI = {
  register: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return client.post('/face/register', form, { headers: { 'Content-Type': 'multipart/form-data' } })
  },
}

// ─── Attendance ────────────────────────────────────────────────
export const attendanceAPI = {
  clockIn: (data: { face_image: string; voice_audio: string; device_id?: string }) =>
    client.post('/attendance/clock-in', data),
  clockOut: (data: { face_image: string; voice_audio: string; device_id?: string }) =>
    client.post('/attendance/clock-out', data),
  getToday: () => client.get('/attendance/today'),
  getHistory: (startDate: string, endDate: string) =>
    client.get('/attendance/history', { params: { start_date: startDate, end_date: endDate } }),
  getEmployeeHistory: (employeeId: string, startDate: string, endDate: string) =>
    client.get(`/attendance/employee/${employeeId}/history`, { params: { start_date: startDate, end_date: endDate } }),
  computeDailySummary: (targetDate?: string) =>
    client.post('/attendance/daily-summary', null, { params: { target_date: targetDate } }),
}

// ─── Departments ───────────────────────────────────────────────
export const departmentAPI = {
  list: () => client.get('/departments/'),
  create: (data: { name: string; description?: string; head_id?: string }) =>
    client.post('/departments/', data),
  update: (id: string, data: { name?: string; description?: string; head_id?: string }) =>
    client.put(`/departments/${id}`, data),
  remove: (id: string) => client.delete(`/departments/${id}`),
}

// ─── Shifts ────────────────────────────────────────────────────
export const shiftAPI = {
  list: () => client.get('/shifts/'),
  create: (data: { name: string; start_time: string; end_time: string; grace_minutes?: number }) =>
    client.post('/shifts/', data),
  update: (id: string, data: Record<string, unknown>) =>
    client.put(`/shifts/${id}`, data),
  remove: (id: string) => client.delete(`/shifts/${id}`),
}

// ─── Employees ─────────────────────────────────────────────────
export const employeeAPI = {
  directory: (departmentId?: string, search?: string) =>
    client.get('/employees/directory', { params: { department_id: departmentId, search } }),
  getMe: () => client.get('/employees/me'),
  get: (id: string) => client.get(`/employees/${id}`),
  create: (data: Record<string, unknown>) => client.post('/employees/', data),
  update: (id: string, data: Record<string, unknown>) => client.put(`/employees/${id}`, data),
}

// ─── Leaves ────────────────────────────────────────────────────
export const leaveAPI = {
  getTypes: () => client.get('/leaves/types'),
  createType: (data: { name: string; days_per_year: number; carry_forward?: boolean; description?: string }) =>
    client.post('/leaves/types', data),
  getMyBalance: (year?: number) => client.get('/leaves/my-balance', { params: { year } }),
  requestLeave: (data: { leave_type_id: string; start_date: string; end_date: string; reason: string }) =>
    client.post('/leaves/request', data),
  getMyRequests: () => client.get('/leaves/my-requests'),
  getPending: () => client.get('/leaves/pending'),
  approve: (id: string, remarks?: string) =>
    client.put(`/leaves/${id}/approve`, { remarks }),
  reject: (id: string, remarks?: string) =>
    client.put(`/leaves/${id}/reject`, { remarks }),
  initializeBalances: (userId: string, year: number) =>
    client.post('/leaves/initialize-balances', null, { params: { user_id: userId, year } }),
}

// ─── Alerts ────────────────────────────────────────────────────
export const alertAPI = {
  list: (params?: { page?: number; page_size?: number; is_read?: boolean; alert_type?: string }) =>
    client.get('/alerts/', { params }),
  unreadCount: () => client.get('/alerts/unread-count'),
  markRead: (id: string) => client.put(`/alerts/${id}/read`),
  markAllRead: () => client.put('/alerts/read-all'),
}

// ─── Reports ───────────────────────────────────────────────────
export const reportAPI = {
  attendance: (startDate: string, endDate: string, departmentId?: string) =>
    client.get('/reports/attendance', { params: { start_date: startDate, end_date: endDate, department_id: departmentId } }),
  overtime: (startDate: string, endDate: string, departmentId?: string) =>
    client.get('/reports/overtime', { params: { start_date: startDate, end_date: endDate, department_id: departmentId } }),
  exportCSV: (startDate: string, endDate: string, departmentId?: string) =>
    client.get('/reports/export/csv', { params: { start_date: startDate, end_date: endDate, department_id: departmentId }, responseType: 'blob' }),
}

// ─── Office Dashboard ──────────────────────────────────────────
export const officeAPI = {
  stats: () => client.get('/office/stats'),
  departmentBreakdown: () => client.get('/office/stats/departments'),
  timeseries: (days?: number) => client.get('/office/timeseries', { params: { days } }),
  liveStatus: () => client.get('/office/live'),
}

// ─── Unified API object ───────────────────────────────────────
const api = {
  auth: authAPI,
  face: faceAPI,
  attendance: attendanceAPI,
  department: departmentAPI,
  shift: shiftAPI,
  employee: employeeAPI,
  leave: leaveAPI,
  alert: alertAPI,
  report: reportAPI,
  office: officeAPI,
  client,
}

export default api

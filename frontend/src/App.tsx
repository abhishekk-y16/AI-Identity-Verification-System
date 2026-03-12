import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './hooks/useAuth'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ClockPage from './pages/ClockPage'
import MyAttendancePage from './pages/MyAttendancePage'
import MyLeavesPage from './pages/MyLeavesPage'
import AdminDashboardPage from './pages/AdminDashboardPage'
import EmployeeDirectoryPage from './pages/EmployeeDirectoryPage'
import AttendanceManagementPage from './pages/AttendanceManagementPage'
import LeaveManagementPage from './pages/LeaveManagementPage'
import DepartmentManagementPage from './pages/DepartmentManagementPage'
import ShiftManagementPage from './pages/ShiftManagementPage'
import ReportsPage from './pages/ReportsPage'
import AlertsPage from './pages/AlertsPage'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="flex items-center justify-center min-h-screen"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" /></div>
  if (!user) return <Navigate to="/login" />
  return <>{children}</>
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="flex items-center justify-center min-h-screen"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" /></div>
  if (!user || (user.role !== 'admin' && user.role !== 'manager')) return <Navigate to="/clock" />
  return <>{children}</>
}

function SmartRedirect() {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" />
  if (user.role === 'admin' || user.role === 'manager') return <Navigate to="/dashboard" />
  return <Navigate to="/clock" />
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          {/* Protected — employee routes */}
          <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route path="/clock" element={<ClockPage />} />
            <Route path="/my-attendance" element={<MyAttendancePage />} />
            <Route path="/my-leaves" element={<MyLeavesPage />} />
            {/* Admin/Manager routes */}
            <Route path="/dashboard" element={<AdminRoute><AdminDashboardPage /></AdminRoute>} />
            <Route path="/employees" element={<AdminRoute><EmployeeDirectoryPage /></AdminRoute>} />
            <Route path="/attendance-mgmt" element={<AdminRoute><AttendanceManagementPage /></AdminRoute>} />
            <Route path="/leave-mgmt" element={<AdminRoute><LeaveManagementPage /></AdminRoute>} />
            <Route path="/departments" element={<AdminRoute><DepartmentManagementPage /></AdminRoute>} />
            <Route path="/shifts" element={<AdminRoute><ShiftManagementPage /></AdminRoute>} />
            <Route path="/reports" element={<AdminRoute><ReportsPage /></AdminRoute>} />
            <Route path="/alerts" element={<AlertsPage />} />
          </Route>
          {/* Smart redirect */}
          <Route path="*" element={<SmartRedirect />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

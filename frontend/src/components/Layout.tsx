import { Outlet, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import {
  Clock, BarChart3, LogOut, Menu, X, Users, Building2,
  CalendarDays, Bell, FileText, Settings, Timer, UserCircle, Home,
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { alertAPI } from '../services/api'

const employeeNav = [
  { path: '/clock', label: 'Clock In/Out', icon: Clock },
  { path: '/my-attendance', label: 'My Attendance', icon: CalendarDays },
  { path: '/my-leaves', label: 'My Leaves', icon: Timer },
]

const adminNav = [
  { path: '/dashboard', label: 'Dashboard', icon: Home },
  { path: '/employees', label: 'Employees', icon: Users },
  { path: '/attendance-mgmt', label: 'Attendance', icon: CalendarDays },
  { path: '/leave-mgmt', label: 'Leaves', icon: Timer },
  { path: '/departments', label: 'Departments', icon: Building2 },
  { path: '/shifts', label: 'Shifts', icon: Clock },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/alerts', label: 'Alerts', icon: Bell },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [unreadAlerts, setUnreadAlerts] = useState(0)

  const isAdmin = user?.role === 'admin' || user?.role === 'manager'
  const navItems = isAdmin ? [...employeeNav, ...adminNav] : employeeNav

  useEffect(() => {
    alertAPI.unreadCount().then(res => setUnreadAlerts(res.data.unread_count)).catch(() => {})
    const interval = setInterval(() => {
      alertAPI.unreadCount().then(res => setUnreadAlerts(res.data.unread_count)).catch(() => {})
    }, 60000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar — desktop */}
      <aside className="hidden lg:flex lg:flex-col w-64 bg-white border-r border-gray-200 fixed inset-y-0 z-40">
        <div className="flex items-center gap-3 px-6 h-16 border-b border-gray-200">
          <Building2 className="h-7 w-7 text-blue-600" />
          <span className="font-bold text-base text-gray-900">Office Attend</span>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon
            const active = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <Icon className="h-5 w-5" />
                {item.label}
                {item.path === '/alerts' && unreadAlerts > 0 && (
                  <span className="ml-auto bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">{unreadAlerts}</span>
                )}
              </Link>
            )
          })}
        </nav>
        <div className="px-4 py-4 border-t border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-700 font-semibold text-sm">{user?.name?.charAt(0).toUpperCase()}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">{user?.name}</p>
              <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
            </div>
            <button onClick={logout} className="text-gray-400 hover:text-red-600 transition-colors" title="Logout">
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen">
        {/* Top bar — mobile */}
        <header className="lg:hidden bg-white border-b border-gray-200 sticky top-0 z-50">
          <div className="flex items-center justify-between px-4 h-14">
            <button onClick={() => setMobileOpen(true)} className="text-gray-500">
              <Menu className="h-6 w-6" />
            </button>
            <span className="font-bold text-gray-900">Office Attend</span>
            <button onClick={logout} className="text-gray-500 hover:text-red-600">
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </header>

        {/* Mobile drawer */}
        {mobileOpen && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <div className="absolute inset-0 bg-black/30" onClick={() => setMobileOpen(false)} />
            <div className="absolute inset-y-0 left-0 w-72 bg-white shadow-xl">
              <div className="flex items-center justify-between px-4 h-14 border-b">
                <span className="font-bold text-gray-900">Menu</span>
                <button onClick={() => setMobileOpen(false)}><X className="h-5 w-5" /></button>
              </div>
              <nav className="px-3 py-4 space-y-1">
                {navItems.map((item) => {
                  const Icon = item.icon
                  const active = location.pathname === item.path
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium ${
                        active ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                      {item.label}
                    </Link>
                  )
                })}
              </nav>
            </div>
          </div>
        )}

        {/* Main content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

# Import all models so SQLAlchemy Base.metadata knows about every table
from app.models.user import User  # noqa: F401
from app.models.department import Department  # noqa: F401
from app.models.shift import Shift  # noqa: F401
from app.models.employee import EmployeeProfile  # noqa: F401
from app.models.attendance import AttendanceRecord, DailySummary  # noqa: F401
from app.models.leave import LeaveType, LeaveBalance, LeaveRequest  # noqa: F401
from app.models.alert import Alert  # noqa: F401

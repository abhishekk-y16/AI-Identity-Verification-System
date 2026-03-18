# AI Identity Verification System (Deep Learning Version)

An advanced full-stack multimodal   biometric identity verification platform powered by deep learning. Supports face recognition, voice authentication, fingerprint matching, liveness detection, deepfake detection, document KYC, behavioral biometrics, fraud detection, and an analytics dashboard. 

---

## Features

| # | Feature | Technology |
|---|---------|-----------|
| 1 | **Face Recognition** | facenet-pytorch (InceptionResnetV1 + MTCNN) |
| 2 | **Voice Authentication** | SpeechBrain ECAPA-TDNN |
| 3 | **Fingerprint Matching** | OpenCV minutiae extraction + spatial histogram |
| 4 | **Liveness Detection** | MobileNetV2 + texture/blink analysis |
| 5 | **Deepfake Detection** | EfficientNet-B0 + frequency analysis |
| 6 | **Document KYC** | EasyOCR + face matching |
| 7 | **Behavioral Biometrics** | Keystroke/mouse dynamics profiling |
| 8 | **Fraud Detection** | Isolation Forest + rule engine |
| 9 | **Biometric Tokenization** | Cancelable biometrics (random projection) |
| 10 | **Analytics Dashboard** | Real-time stats, charts, logs |

## Architecture 

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   React SPA  │────▶│  FastAPI      │────▶│ PostgreSQL  │
│  (TailwindCSS│     │  Backend      │     │ (SQLAlchemy)│
│   + Vite)    │     │              │────▶│ FAISS       │
└──────────────┘     │  ML Models:  │     │ (Vectors)   │
                     │  PyTorch     │     └─────────────┘
                     └──────────────┘
```

## Tech Stack 

**Backend:** Python 3.11, FastAPI, SQLAlchemy (async), PyTorch, FAISS, scikit-learn  
**Frontend:** React 18, TypeScript, Vite, TailwindCSS, Recharts  
**Infrastructure:** Docker Compose, PostgreSQL 16, Redis 7, Nginx  
**Auth:** JWT (python-jose), bcrypt (passlib)

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) Python 3.11+ and Node.js 20+ for local development

### 1. Clone & Configure

```bash
cp .env.example .env
# Edit .env — at minimum change SECRET_KEY for production
```

### 2. Run with Docker Compose 

```bash
docker-compose up --build
```

This starts:
- **PostgreSQL** on port 5432
- **Redis** on port 6379
- **Backend API** on port 8000
- **Frontend** on port 3000

### 3. Access the App

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |

---

## Local Development (without Docker)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Requires PostgreSQL running locally (update `DATABASE_URL` in `.env`).

### Frontend 

```bash
cd frontend
npm install
npm run dev
```

Vite dev server starts on http://localhost:5173 with API proxy to :8000.

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Login, get JWT |
| GET | `/api/auth/me` | Current user info |

### Biometric Enrollment
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/face/register` | Enroll face |
| POST | `/api/voice/register` | Enroll voice |
| POST | `/api/fingerprint/register` | Enroll fingerprint |

### Verification
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/face/verify` | Face verification |
| POST | `/api/face/identify` | 1:N face identification |
| POST | `/api/voice/verify` | Voice verification |
| POST | `/api/fingerprint/verify` | Fingerprint verification |
| POST | `/api/verify/full` | Full multimodal verification |

### Detection & Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/liveness/check` | Image liveness check |
| POST | `/api/liveness/check-video` | Video liveness check |
| POST | `/api/deepfake/detect` | Deepfake detection (image) |
| POST | `/api/deepfake/detect-video` | Deepfake detection (video) |
| POST | `/api/document/verify` | Document KYC |
| POST | `/api/fraud/check` | Fraud risk assessment |

### Dashboard (Admin)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/stats` | Platform statistics |
| GET | `/api/dashboard/logs` | Verification logs |
| GET | `/api/dashboard/users` | User list |
| GET | `/api/dashboard/alerts` | Fraud alerts |
| GET | `/api/dashboard/timeseries` | Time series data |

---

## Running Tests

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest -v
```

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py            # Settings & thresholds
│   │   ├── database.py          # Async SQLAlchemy setup
│   │   ├── main.py              # FastAPI app & lifespan
│   │   ├── models/              # ORM models (User, VerificationLog)
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── ml/                  # ML model wrappers
│   │   │   ├── face_model.py    # MTCNN + InceptionResnetV1
│   │   │   ├── voice_model.py   # ECAPA-TDNN
│   │   │   ├── fingerprint_model.py  # OpenCV minutiae
│   │   │   ├── liveness_model.py     # MobileNetV2 + blink
│   │   │   ├── deepfake_model.py     # EfficientNet-B0
│   │   │   └── embeddings.py    # FAISS index manager
│   │   ├── services/            # Business logic layer
│   │   ├── routers/             # API route handlers
│   │   └── utils/               # Security, image, audio helpers
│   ├── tests/                   # pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/               # Route pages
│   │   ├── hooks/               # Auth context
│   │   ├── services/            # API client
│   │   ├── App.tsx              # Router
│   │   └── main.tsx             # Entry point
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Security Features

- **JWT authentication** with role-based access control (user/admin)
- **Cancelable biometrics** — biometric templates are protected via random orthogonal projection matrices; templates can be revoked and reissued without re-enrollment
- **bcrypt password hashing** via passlib
- **Input validation** on all endpoints via Pydantic
- **CORS** configured for allowed origins only
- **File type & size validation** for all uploads

## License 

MIT

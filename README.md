# AI Resume Screening & Semantic Job Matching Web Application

An enterprise-grade, responsive AI Resume Screening, Job Matching, and Career Studio platform. Built with **FastAPI** on the backend and **React + Tailwind CSS** on the frontend, featuring a signature purple and lavender theme, Sentence-BERT vector matching, trained ML category prediction, factual AI resume tailoring, and persistent multi-user storage.

---

## 🌟 Key Features

1. **Purple & Lavender Design System**:
   - Signature royal purple and soft lavender palette (`#30236f`, `#5938d5`, `#7650ed`, `#ece8ff`, `#f5f6ff`).
   - Sleek responsive sidebar navigation, modern cards with subtle elevation, Lucide icons, and real-time status pills.

2. **Resume Builder with Live Preview & Multi-Format Export**:
   - Multi-section forms for **Personal Details**, **Professional Summary**, **Skills**, **Work Experience**, **Internships**, **Projects**, **Education**, **Certifications**, and **Achievements**.
   - Real-time side-by-side live resume preview.
   - One-click professional **DOCX** (`python-docx`) and **PDF** (`ReportLab`) downloads styled with royal purple headers and clean typography.
   - Persistent cloud profile saving to user accounts.

3. **Factual AI Resume Tailoring**:
   - Input any target job description to dynamically reorganize resume competencies, prioritizing matched job keywords.
   - Rewrites and sharpens experience bullets with strong action verbs (e.g. *Engineered*, *Spearheaded*, *Optimized*) and aligns professional summaries.
   - **Strict Zero-Hallucination Guardrail**: Never fabricates or invents fake jobs, degrees, dates, or false metrics.
   - **Configurable Provider**: Supports Google Gemini AI (`gemini-1.5-flash`) via `GEMINI_API_KEY` or the in-app Settings modal; falls back seamlessly to an intelligent **Deterministic Rule-Based NLP Optimizer** (100% offline). The active provider is transparently reported in the UI.

4. **Sentence-BERT Semantic Job Matching**:
   - Accepts **PDF**, **DOCX**, and **TXT** resume uploads or uses the active built resume profile.
   - Vectorizes resume content using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings).
   - Instantaneously ranks opportunities across a curated catalog of **25,000+ jobs** with sub-30ms cosine similarity latency.
   - Shows similarity percentage badges, matched skills badges (green pills), and missing skills badges (rose pills).

5. **Interactive Skill Gap Analyzer**:
   - Side-by-side requirement coverage meter comparing resume skills against job requirements.
   - Clear categorized breakdown: Verified Matched Skills, Critical Missing Skills, and Bonus Candidate Strengths.
   - Prominent disclaimer highlighting that keyword extraction is pattern/dictionary-approximate.

6. **Resume Industry Category Prediction**:
   - Reuses the trained scikit-learn TF-IDF + Logistic Regression pipeline (`models/category_classifier.pkl`).
   - Classifies any resume into one of 24 industry sectors (e.g., `ENGINEERING`, `HR`, `FINANCE`, `INFORMATION-TECHNOLOGY`, `SALES`, etc.) with top-5 confidence probability distribution bars.

7. **Search, Facet Filters, Saved Jobs, History & CSV Export**:
   - Multi-facet filters: Keyword search, Role/Category, Location, Work Type (Full-time, Intern, Remote, Temporary), and Min Similarity threshold slider.
   - Bookmarking / Saved Jobs management with custom notes.
   - Automatic matching session history tracking.
   - Single-click CSV export of ranked job matches.

8. **Secure Authentication & Persistent Storage**:
   - SQLite relational database with SQLAlchemy (`data/app_database.db`).
   - Cryptographic password hashing using **PBKDF2-HMAC-SHA256** with 100,000 rounds and random salt (zero plaintext passwords).
   - Bearer JWT token authentication.

---

## 🏛️ System Architecture

```
AI_Resume_Matcher/
├── backend/
│   ├── core/
│   │   ├── config.py             # Settings, paths, and secrets
│   │   └── security.py           # Salted password hashing & JWT tokens
│   ├── database/
│   │   ├── models.py             # SQLAlchemy models (User, Profile, SavedJob, MatchHistory)
│   │   └── session.py            # SQLite engine & database sessions
│   ├── services/
│   │   ├── model_service.py      # Sentence-BERT singleton & Category Classifier
│   │   ├── job_service.py        # 25,000-job catalog search, filters & vector similarity
│   │   ├── skill_service.py      # Skill taxonomy & gap analysis
│   │   ├── tailoring_service.py  # Factual AI tailoring (Gemini + Rule-based NLP fallback)
│   │   └── resume_export_service.py # DOCX and PDF document generators
│   ├── routers/
│   │   ├── auth.py               # /api/auth (register, login, me)
│   │   ├── resume.py             # /api/resume (upload, category, tailor, export)
│   │   ├── jobs.py               # /api/jobs (match, filters, export-csv)
│   │   ├── saved_jobs.py         # /api/saved-jobs (CRUD)
│   │   └── history.py            # /api/history (match audit sessions)
│   └── main.py                   # FastAPI app, CORS, lifespan & /api/health
│
├── frontend/
│   ├── src/
│   │   ├── components/           # Sidebar, Header, AuthModal, SettingsModal, Toast
│   │   ├── views/                # Dashboard, Builder, Tailoring, Matching, SkillGap, Classifier, Saved, History
│   │   ├── services/             # Axios API client with JWT interceptor
│   │   ├── App.jsx               # Navigation router & state coordinator
│   │   └── index.css             # Tailwind CSS & purple/lavender styling
│   ├── package.json
│   └── vite.config.js            # Vite config with /api proxy to port 8000
│
├── data/
│   ├── processed/
│   │   ├── cleaned_resumes.csv   # Preserved original resume dataset
│   │   ├── cleaned_jobs.csv      # Preserved original jobs dataset
│   │   ├── jobs_catalog.csv      # Enriched 25,000-job catalog with Location & Work Type
│   │   └── job_embeddings.npy    # Precomputed Sentence-BERT embeddings
│   └── app_database.db           # SQLite multi-user database
│
├── models/
│   └── category_classifier.pkl   # Preserved trained 24-category scikit-learn model
│
├── scripts/
│   ├── prepare_catalog.py        # Catalog extractor and vector pre-encoder
│   └── test_backend.py           # Automated backend test suite
│
├── run_backend.bat               # Starts FastAPI backend (Port 8000)
├── run_frontend.bat              # Starts React frontend (Port 5173)
├── run_app.bat                   # Starts both simultaneously
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.12+**
- **Node.js v18+** & **npm**

### 2. Launch the Application

#### Option A: One-Click Startup (Windows)
Double-click `run_app.bat` or run in PowerShell:
```powershell
.\run_app.bat
```
This automatically launches:
- **Backend API**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **React Frontend**: `http://localhost:5173`

#### Option B: Individual Terminals

**Terminal 1 (Backend)**:
```powershell
.\venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend)**:
```powershell
cd frontend
npm run dev
```

Visit **`http://localhost:5173`** in your browser.

---

## 🧪 Automated Testing

To run the complete automated API and ML verification suite:
```powershell
.\venv\Scripts\python.exe scripts\test_backend.py
```
This tests:
1. Health check & model pre-warm
2. User registration, login, and JWT verification
3. Industry category prediction (`models/category_classifier.pkl`)
4. Skill gap analysis and approximate notice
5. Sentence-BERT semantic job matching and multi-facet filtering
6. AI resume tailoring and factual preservation
7. DOCX and PDF resume exports
8. Saved jobs bookmarking CRUD operations

---

## 🔑 Environment Variables (`.env`)

Create a `.env` file in the project root (or copy `.env.example`):
```env
# Secret key for JWT session tokens
SECRET_KEY=super-secret-purple-lavender-ai-key-2026-secure
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Database URL
DATABASE_URL=sqlite:///./data/app_database.db

# Optional Google Gemini API Key for LLM resume tailoring
# If omitted, the platform uses its built-in rule-based NLP optimizer (100% offline, fact-preserving)
GEMINI_API_KEY=
```

You can also set or change the Gemini API key anytime via the in-app **Settings & AI Engine** modal without restarting the server.

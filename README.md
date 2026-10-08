# AI-Powered 6G RIS Platform — JCAS Dashboard
### v2.4.1 | Resume-Ready Research Project

---

## What This Demonstrates (Resume Bullets)

- **FastAPI backend** — async REST API with auto-generated OpenAPI docs
- **SQLite database** — persistent simulation result storage
- **SQLAlchemy ORM** — typed database models, session management
- **AI-based SNR prediction** — scikit-learn Random Forest with physics-informed features
- **RIS optimization** — greedy & hill-climbing phase search algorithms
- **Simulation engine** — full THz path-loss, SINR, WSR, outage, Doppler physics
- **Dashboard analytics** — aggregate KPIs over all simulation runs
- **Frontend integration** — live dashboard UI communicates with FastAPI backend
- **Research-oriented project structure** — clean separation of concerns

---

## Project Structure

```
ai_6g_ris_platform/
├── main.py            ← FastAPI app — all endpoints
├── database.py        ← SQLAlchemy + SQLite setup
├── models.py          ← ORM models (SimulationResult, NetworkState)
├── schemas.py         ← Pydantic request/response schemas
├── simulation.py      ← Physics engine (THz, SINR, WSR, outage, Doppler)
├── predictor.py       ← scikit-learn SNR predictor (train + inference)
├── optimizer.py       ← RIS phase optimization (greedy, hill-climbing)
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── frontend/
    ├── index.html     ← JCAS Dashboard UI
    ├── style.css      ← Dashboard styling
    └── script.js      ← API integration (connects UI ↔ backend)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Pre-train the ML model
```bash
python predictor.py
```
This generates 3,000 synthetic samples and trains + saves `snr_predictor.pkl`.
If skipped, the model auto-trains on first prediction request.

### 3. Start the backend
```bash
python main.py
```
or with hot-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the dashboard
Visit **http://localhost:8000** in your browser.

### 5. Explore the API docs
- Swagger UI: **http://localhost:8000/api/docs**
- ReDoc:       **http://localhost:8000/api/redoc**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/simulate` | Full physics simulation + AI + RIS optimize |
| `POST` | `/api/predict` | ML-only SNR prediction |
| `POST` | `/api/optimize` | RIS phase optimization |
| `GET`  | `/api/analytics` | Aggregate stats over all runs |
| `GET`  | `/api/simulations` | List saved results (paginated) |
| `GET`  | `/api/simulations/{id}` | Single result by ID |
| `DELETE` | `/api/simulations/{id}` | Delete a result |
| `GET`  | `/api/dataset/generate?n=100` | Generate synthetic training data |
| `POST` | `/api/model/retrain?n_samples=2000` | Retrain ML model |
| `GET`  | `/api/health` | Health check |

---

## Example API Call

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "velocity": 45.0,
    "ris_elements": 64,
    "tx_power_dbm": 30.0,
    "alpha": 0.6,
    "frequency_thz": 0.3,
    "num_users": 4
  }'
```

**Response:**
```json
{
  "id": 1,
  "snr_db": 18.4,
  "wsr_mbps": 24.7,
  "outage_pct": 5.3,
  "doppler_hz": 833.0,
  "rl_reward": 12.4,
  "predicted_snr": 19.1,
  "optimized_phase": "[0.785, 1.571, 2.356, ...]"
}
```

---

## Physics Models

| Model | Formula |
|-------|---------|
| THz Path Loss | `L(d,f) = L_spread + L_abs` |
| RIS Channel | `h_eff = h_d + Σ(h_r · Φ · g)` |
| Weighted Sum Rate | `WSR = α·Rc + (1-α)·Rs` |
| Doppler Shift | `f_d = v·f_c / c` |
| SINR | `SINR = P|h|² / (I + N₀)` |
| Outage Probability | `P_out = 1 - exp(-γ_th / SINR)` |
| RL Reward | `R = WSR - 2·P_out + 0.02·N_RIS` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Database | SQLite (swap to PostgreSQL via `DATABASE_URL` env var) |
| ORM | SQLAlchemy |
| AI/ML | scikit-learn (RandomForest), joblib |
| Validation | Pydantic v2 |
| Frontend | HTML5, CSS3, JavaScript (Chart.js) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./ris_platform.db` | DB connection string |
| `PORT` | `8000` | Server port |

---


"""
main.py
AI-Powered 6G RIS Platform — FastAPI Backend
JCAS Dashboard | v2.4.1

Endpoints:
  POST /api/simulate          — run physics simulation
  POST /api/predict           — ML SNR prediction
  POST /api/optimize          — RIS phase optimization
  GET  /api/analytics         — DB aggregation stats
  GET  /api/simulations       — list saved results
  GET  /api/simulations/{id}  — single result
  DELETE /api/simulations/{id}— delete result
  GET  /api/dataset/generate  — generate synthetic dataset
  POST /api/model/retrain     — retrain ML model
  GET  /api/health            — health check
  GET  /                      — serve frontend
"""

import json
import os
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import uvicorn

from database import engine, get_db, Base
from models import SimulationResult, NetworkState
from schemas import (
    SimulationRequest, SimulationResponse,
    PredictRequest,    PredictResponse,
    OptimizeRequest,   OptimizeResponse,
    AnalyticsResponse, HealthResponse,
)
from simulation import run_simulation, generate_dataset
from predictor import SNRPredictor
from optimizer import RISOptimizer

# ── Bootstrap ─────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

predictor = SNRPredictor()   # singleton — lazy-loads model
optimizer = RISOptimizer()

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered 6G RIS Platform",
    description=(
        "JCAS Dashboard — RIS Optimization, AI-based SNR Prediction, "
        "Simulation Engine, Dashboard Analytics. "
        "Stack: FastAPI · SQLite · SQLAlchemy · scikit-learn"
    ),
    version="2.4.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend (if built) ─────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/static", StaticFiles(directory=_frontend), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(os.path.join(_frontend, "index.html"))


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/simulate", response_model=SimulationResponse, tags=["Simulation"])
def simulate(req: SimulationRequest, db: Session = Depends(get_db)):
    """
    Run a full RIS-assisted 6G JCAS physics simulation.

    Computes: SNR, SINR, WSR, outage probability, Doppler shift, RL reward.
    Also runs AI SNR prediction and RIS phase optimization automatically.
    Saves result to SQLite.
    """
    # Physics simulation
    result = run_simulation(
        velocity      = req.velocity,
        ris_elements  = req.ris_elements,
        tx_power_dbm  = req.tx_power_dbm,
        alpha         = req.alpha,
        frequency_thz = req.frequency_thz,
        num_users     = req.num_users,
    )

    # AI prediction
    pred = predictor.predict(
        velocity      = req.velocity,
        ris_elements  = req.ris_elements,
        tx_power_dbm  = req.tx_power_dbm,
        frequency_thz = req.frequency_thz,
        num_users     = req.num_users,
    )

    # RIS optimization
    opt = optimizer.optimize(
        ris_elements  = req.ris_elements,
        target_snr_db = result.snr_db + 3,
        phase_bits    = 4,
        tx_power_dbm  = req.tx_power_dbm,
        freq_thz      = req.frequency_thz,
    )

    # Persist to DB
    db_row = SimulationResult(
        velocity        = req.velocity,
        ris_elements    = req.ris_elements,
        tx_power_dbm    = req.tx_power_dbm,
        alpha           = req.alpha,
        frequency_thz   = req.frequency_thz,
        num_users       = req.num_users,
        snr_db          = result.snr_db,
        wsr_mbps        = result.wsr_mbps,
        outage_pct      = result.outage_pct,
        doppler_hz      = result.doppler_hz,
        rl_reward       = result.rl_reward,
        predicted_snr   = pred["predicted_snr_db"],
        optimized_phase = json.dumps(opt["optimized_phases"][:8]),   # store first 8
    )
    db.add(db_row)
    db.commit()
    db.refresh(db_row)
    return db_row


# ═══════════════════════════════════════════════════════════════════════════
# AI PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/predict", response_model=PredictResponse, tags=["AI Engine"])
def predict_snr(req: PredictRequest):
    """
    ML-based SNR prediction using a trained Random Forest model.
    Features include physics-informed engineered inputs (Doppler, RIS gain, path-loss proxy).
    """
    result = predictor.predict(
        velocity      = req.velocity,
        ris_elements  = req.ris_elements,
        tx_power_dbm  = req.tx_power_dbm,
        frequency_thz = req.frequency_thz,
        num_users     = req.num_users,
    )
    return result


@app.post("/api/model/retrain", tags=["AI Engine"])
def retrain_model(n_samples: int = Query(2000, ge=500, le=10000)):
    """
    Retrain the SNR prediction model on freshly generated simulation data.
    Replaces the saved model file on disk.
    """
    metrics = predictor.retrain(n_samples=n_samples)
    return {"status": "retrained", "metrics": metrics}


# ═══════════════════════════════════════════════════════════════════════════
# RIS OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/optimize", response_model=OptimizeResponse, tags=["RIS Optimizer"])
def optimize_ris(req: OptimizeRequest):
    """
    Optimize RIS phase configuration to maximise SNR.

    Strategy:
      - ≤64 elements  → greedy element-wise search (optimal, O(N·2^b))
      - >64 elements  → random-restart hill climbing (scalable)
    """
    result = optimizer.optimize(
        ris_elements  = req.ris_elements,
        target_snr_db = req.target_snr_db,
        phase_bits    = req.phase_bits,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICS & HISTORY
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def analytics(db: Session = Depends(get_db)):
    """
    Aggregate statistics over all saved simulation runs.
    """
    rows = db.query(SimulationResult).all()
    if not rows:
        return AnalyticsResponse(
            total_simulations=0, avg_snr_db=0.0, avg_wsr_mbps=0.0,
            avg_outage_pct=0.0, best_snr_db=0.0, best_wsr_mbps=0.0,
        )
    snrs  = [r.snr_db  for r in rows if r.snr_db  is not None]
    wsrs  = [r.wsr_mbps for r in rows if r.wsr_mbps is not None]
    outs  = [r.outage_pct for r in rows if r.outage_pct is not None]
    return AnalyticsResponse(
        total_simulations = len(rows),
        avg_snr_db        = round(sum(snrs) / len(snrs), 3)  if snrs else 0,
        avg_wsr_mbps      = round(sum(wsrs) / len(wsrs), 3)  if wsrs else 0,
        avg_outage_pct    = round(sum(outs) / len(outs), 3)  if outs else 0,
        best_snr_db       = round(max(snrs), 3)              if snrs else 0,
        best_wsr_mbps     = round(max(wsrs), 3)              if wsrs else 0,
    )


@app.get("/api/simulations", response_model=List[SimulationResponse], tags=["Analytics"])
def list_simulations(
    limit:  int = Query(50,  ge=1,  le=500),
    offset: int = Query(0,   ge=0),
    db: Session = Depends(get_db),
):
    """List saved simulation results (paginated)."""
    return (db.query(SimulationResult)
              .order_by(SimulationResult.id.desc())
              .offset(offset).limit(limit).all())


@app.get("/api/simulations/{sim_id}", response_model=SimulationResponse, tags=["Analytics"])
def get_simulation(sim_id: int, db: Session = Depends(get_db)):
    """Retrieve a single simulation result by ID."""
    row = db.query(SimulationResult).filter(SimulationResult.id == sim_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    return row


@app.delete("/api/simulations/{sim_id}", tags=["Analytics"])
def delete_simulation(sim_id: int, db: Session = Depends(get_db)):
    """Delete a simulation result by ID."""
    row = db.query(SimulationResult).filter(SimulationResult.id == sim_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    db.delete(row)
    db.commit()
    return {"deleted": sim_id}


# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/dataset/generate", tags=["Simulation"])
def dataset_generate(n: int = Query(100, ge=10, le=5000)):
    """
    Generate n synthetic simulation samples.
    Returns first 10 as preview; full count in metadata.
    Useful for ML training data export.
    """
    data = generate_dataset(n_samples=n)
    return {
        "generated": len(data),
        "preview":   data[:10],
        "note":      "Use POST /api/model/retrain to retrain the ML model with fresh data.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health(db: Session = Depends(get_db)):
    """Health check — verifies DB connectivity."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return HealthResponse(
        status="online", version="2.4.1",
        platform="AI-6G-RIS", db=db_status,
    )


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
schemas.py
Pydantic schemas — request validation & response serialization
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Request Schemas ───────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    velocity:      float = Field(45.0,  ge=0,   le=120,  description="UE velocity km/h")
    ris_elements:  int   = Field(64,    ge=16,  le=256,  description="Number of RIS elements")
    tx_power_dbm:  float = Field(30.0,  ge=0,   le=50,   description="Transmit power dBm")
    alpha:         float = Field(0.6,   ge=0.0, le=1.0,  description="JCAS tradeoff α (0=sensing, 1=comm)")
    frequency_thz: float = Field(0.3,   ge=0.1, le=1.0,  description="Carrier frequency THz")
    num_users:     int   = Field(4,     ge=1,   le=16,   description="Number of active users")

    class Config:
        json_schema_extra = {
            "example": {
                "velocity": 45.0,
                "ris_elements": 64,
                "tx_power_dbm": 30.0,
                "alpha": 0.6,
                "frequency_thz": 0.3,
                "num_users": 4
            }
        }


class PredictRequest(BaseModel):
    velocity:      float = Field(..., ge=0,  le=120)
    ris_elements:  int   = Field(..., ge=16, le=256)
    tx_power_dbm:  float = Field(..., ge=0,  le=50)
    frequency_thz: float = Field(..., ge=0.1, le=1.0)
    num_users:     int   = Field(..., ge=1,  le=16)


class OptimizeRequest(BaseModel):
    ris_elements:  int   = Field(64,  ge=16, le=256)
    target_snr_db: float = Field(20.0, description="Target SNR in dB")
    phase_bits:    int   = Field(4,   ge=1,  le=8,   description="Phase quantization bits")


# ── Response Schemas ──────────────────────────────────────────────────────

class SimulationResponse(BaseModel):
    id:             int
    velocity:       float
    ris_elements:   int
    tx_power_dbm:   float
    alpha:          float
    frequency_thz:  float
    num_users:      int
    snr_db:         float
    wsr_mbps:       float
    outage_pct:     float
    doppler_hz:     float
    rl_reward:      float
    predicted_snr:  Optional[float] = None
    optimized_phase: Optional[str]  = None
    created_at:     Optional[datetime] = None

    class Config:
        from_attributes = True


class PredictResponse(BaseModel):
    predicted_snr_db:   float
    confidence:         float
    model:              str


class OptimizeResponse(BaseModel):
    ris_elements:       int
    optimized_phases:   List[float]
    estimated_snr_gain: float
    phase_bits:         int
    strategy:           str


class AnalyticsResponse(BaseModel):
    total_simulations:  int
    avg_snr_db:         float
    avg_wsr_mbps:       float
    avg_outage_pct:     float
    best_snr_db:        float
    best_wsr_mbps:      float


class HealthResponse(BaseModel):
    status:   str
    version:  str
    platform: str
    db:       str

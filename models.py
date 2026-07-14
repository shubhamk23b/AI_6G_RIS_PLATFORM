"""
models.py
SQLAlchemy ORM models — Simulation table
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from database import Base


class SimulationResult(Base):
    """Stores every simulation run result."""
    __tablename__ = "simulation_results"

    id           = Column(Integer, primary_key=True, index=True)
    velocity     = Column(Float,   nullable=False, comment="UE velocity km/h")
    ris_elements = Column(Integer, nullable=False, comment="Number of RIS elements")
    tx_power_dbm = Column(Float,   nullable=False, comment="Transmit power dBm")
    alpha        = Column(Float,   nullable=False, comment="JCAS tradeoff factor 0-1")
    frequency_thz= Column(Float,   nullable=False, comment="Carrier frequency THz")
    num_users    = Column(Integer, nullable=False, comment="Active UE count")
    snr_db       = Column(Float,   nullable=True,  comment="Computed SNR dB")
    wsr_mbps     = Column(Float,   nullable=True,  comment="Weighted sum rate Mbps")
    outage_pct   = Column(Float,   nullable=True,  comment="Outage probability %")
    doppler_hz   = Column(Float,   nullable=True,  comment="Doppler shift Hz")
    rl_reward    = Column(Float,   nullable=True,  comment="RL agent reward signal")
    predicted_snr= Column(Float,   nullable=True,  comment="ML-predicted SNR dB")
    optimized_phase= Column(String, nullable=True, comment="JSON: optimized RIS phases")
    created_at   = Column(DateTime(timezone=True), server_default=func.now())


class NetworkState(Base):
    """Live network state snapshots for analytics."""
    __tablename__ = "network_states"

    id           = Column(Integer, primary_key=True, index=True)
    snr_db       = Column(Float, nullable=False)
    sinr_db      = Column(Float, nullable=False)
    wsr_mbps     = Column(Float, nullable=False)
    outage_pct   = Column(Float, nullable=False)
    ris_gain_db  = Column(Float, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

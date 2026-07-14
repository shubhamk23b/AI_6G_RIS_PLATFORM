"""
simulation.py
Wireless Physics Simulation — RIS-assisted 6G JCAS
Implements: THz path loss, RIS channel, SINR, WSR, outage probability, Doppler
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Any

# ── Constants ─────────────────────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8          # m/s
BOLTZMANN      = 1.38e-23     # J/K
TEMPERATURE_K  = 290.0        # Kelvin
NOISE_FIGURE   = 7.0          # dB
BANDWIDTH_HZ   = 1e9          # 1 GHz bandwidth
DISTANCE_M     = 100.0        # BS-to-UE distance (metres)


@dataclass
class SimResult:
    snr_db:      float
    wsr_mbps:    float
    outage_pct:  float
    doppler_hz:  float
    rl_reward:   float
    sinr_db:     float
    ris_gain_db: float
    path_loss_db: float


# ── Helper maths ──────────────────────────────────────────────────────────

def to_db(linear: float) -> float:
    """Convert linear power to dB."""
    return 10.0 * math.log10(max(linear, 1e-20))


def from_db(db: float) -> float:
    """Convert dB to linear power."""
    return 10.0 ** (db / 10.0)


def thz_path_loss(distance_m: float, freq_thz: float) -> float:
    """
    THz path loss = free-space spreading loss + molecular absorption loss.
    L(d,f) = L_spread + L_abs  [dB]
    """
    freq_hz  = freq_thz * 1e12
    # Free-space path loss (Friis)
    l_spread = 20 * math.log10(4 * math.pi * distance_m * freq_hz / SPEED_OF_LIGHT)
    # Molecular absorption (simplified water vapour model)
    l_abs    = 10 * math.log10(math.exp(0.002 * freq_thz * distance_m))
    return l_spread + l_abs


def thermal_noise_dbm() -> float:
    """Thermal noise power N0·B in dBm."""
    n0 = BOLTZMANN * TEMPERATURE_K          # W/Hz
    noise_w = n0 * BANDWIDTH_HZ             # W
    noise_dbm = 10 * math.log10(noise_w) + 30 + NOISE_FIGURE
    return noise_dbm


def ris_array_gain_db(n_elements: int) -> float:
    """
    RIS beamforming gain ≈ N² for coherent combination.
    Returns gain in dB.
    """
    return 20 * math.log10(n_elements)


def doppler_shift_hz(velocity_kmh: float, freq_thz: float) -> float:
    """
    Doppler shift: f_d = (v/c) · f_c
    velocity in km/h converted to m/s.
    """
    v_ms = velocity_kmh / 3.6
    return (v_ms * freq_thz * 1e12) / SPEED_OF_LIGHT


def compute_sinr(tx_power_dbm: float,
                 path_loss_db: float,
                 ris_gain_db:  float,
                 num_users:    int,
                 noise_dbm:    float) -> float:
    """
    SINR = P · |h_eff|² / (I + N0)
    h_eff = direct + RIS-reflected channel
    Returns SINR as linear ratio.
    """
    # Received signal power (dBm → linear W)
    rx_power_dbm = tx_power_dbm - path_loss_db + ris_gain_db
    rx_power_w   = from_db(rx_power_dbm - 30)   # dBm → W

    # Co-channel interference (other users, simplified)
    interference_w = (num_users - 1) * from_db(tx_power_dbm - 30) * 1e-4

    # Noise
    noise_w = from_db(noise_dbm - 30)

    sinr = rx_power_w / (interference_w + noise_w)
    return max(sinr, 1e-10)


def weighted_sum_rate(alpha: float, sinr: float,
                      velocity_kmh: float, ris_elements: int) -> float:
    """
    WSR = α · R_c + (1-α) · R_s   [Mbps]
    R_c = Shannon capacity (comm)
    R_s = sensing information rate (velocity/RIS dependent)
    """
    r_comm    = math.log2(1 + sinr) * (BANDWIDTH_HZ / 1e6)          # Mbps
    r_sensing = max(0.1, 2.5 - velocity_kmh * 0.015 + ris_elements * 0.01)
    return alpha * r_comm + (1 - alpha) * r_sensing


def outage_probability(sinr: float, sinr_threshold_db: float = 10.0) -> float:
    """
    P_out = P(R < R_th) ≈ 1 - exp(-γ_th / SINR)  [%]
    Rayleigh fading approximation.
    """
    gamma_th = from_db(sinr_threshold_db)
    p_out    = (1 - math.exp(-gamma_th / sinr)) * 100
    return round(max(0.0, min(p_out, 100.0)), 4)


def rl_reward(wsr: float, outage_pct: float, ris_elements: int) -> float:
    """
    Simple RL reward function used by PPO/DQN agents:
    R = WSR - 2·P_out/100 + 0.02·N_RIS
    """
    return round(wsr - 2 * (outage_pct / 100) + ris_elements * 0.02, 4)


# ── Main simulation entry point ───────────────────────────────────────────

def run_simulation(velocity: float,
                   ris_elements: int,
                   tx_power_dbm: float,
                   alpha: float,
                   frequency_thz: float,
                   num_users: int,
                   add_noise: bool = True) -> SimResult:
    """
    Full RIS-assisted 6G JCAS simulation.

    Parameters
    ----------
    velocity       : UE speed in km/h
    ris_elements   : number of RIS reflective elements (16–256)
    tx_power_dbm   : base station transmit power in dBm
    alpha          : JCAS tradeoff coefficient (0=sensing only, 1=comm only)
    frequency_thz  : carrier frequency in THz (0.1–1.0)
    num_users      : number of active user equipments
    add_noise      : inject small random channel noise for realism

    Returns
    -------
    SimResult dataclass with all KPIs
    """
    # 1) Path loss
    pl_db = thz_path_loss(DISTANCE_M, frequency_thz)

    # 2) RIS gain
    ris_gain = ris_array_gain_db(ris_elements)

    # 3) Noise floor
    noise_dbm = thermal_noise_dbm()

    # 4) SINR
    sinr = compute_sinr(tx_power_dbm, pl_db, ris_gain, num_users, noise_dbm)

    # Optional: small Rayleigh fade noise
    if add_noise:
        fade = 0.8 + random.random() * 0.4   # ±20%
        sinr *= fade

    snr_db  = to_db(sinr)
    sinr_db = snr_db   # simplified (single-user equivalent)

    # 5) WSR
    wsr = weighted_sum_rate(alpha, sinr, velocity, ris_elements)

    # 6) Outage
    outage = outage_probability(sinr)

    # 7) Doppler
    dop = doppler_shift_hz(velocity, frequency_thz)

    # 8) RL reward
    reward = rl_reward(wsr, outage, ris_elements)

    return SimResult(
        snr_db       = round(snr_db, 3),
        wsr_mbps     = round(wsr, 3),
        outage_pct   = outage,
        doppler_hz   = round(dop, 2),
        rl_reward    = reward,
        sinr_db      = round(sinr_db, 3),
        ris_gain_db  = round(ris_gain, 3),
        path_loss_db = round(pl_db, 3),
    )


def generate_dataset(n_samples: int = 1000) -> list[Dict[str, Any]]:
    """
    Generate synthetic CSI / simulation dataset for ML training.
    Returns list of dicts ready for DB insertion or CSV export.
    """
    dataset = []
    for _ in range(n_samples):
        vel   = random.uniform(0, 120)
        ris   = random.choice([16, 32, 64, 128, 256])
        pwr   = random.uniform(10, 46)
        alpha = random.uniform(0.0, 1.0)
        freq  = random.uniform(0.1, 1.0)
        users = random.randint(1, 8)

        r = run_simulation(vel, ris, pwr, alpha, freq, users, add_noise=True)
        dataset.append({
            "velocity": vel, "ris_elements": ris,
            "tx_power_dbm": pwr, "alpha": alpha,
            "frequency_thz": freq, "num_users": users,
            "snr_db": r.snr_db, "wsr_mbps": r.wsr_mbps,
            "outage_pct": r.outage_pct, "doppler_hz": r.doppler_hz,
            "rl_reward": r.rl_reward,
        })
    return dataset

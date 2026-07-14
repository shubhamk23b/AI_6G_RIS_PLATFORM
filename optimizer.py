"""
optimizer.py
RIS Phase Optimizer — maximises SNR / WSR via greedy + gradient search.
Supports: greedy search, random hill-climbing, codebook search.
"""

import math
import random
import numpy as np
from typing import List, Dict, Any

SPEED_OF_LIGHT = 3e8


# ── Phase quantization ────────────────────────────────────────────────────

def quantize_phases(phases: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantize continuous phases [0, 2π) to 2^bits discrete levels.
    """
    levels     = 2 ** bits
    step       = 2 * math.pi / levels
    quantized  = np.round(phases / step) * step
    return quantized % (2 * math.pi)


def phase_to_codebook(bits: int) -> np.ndarray:
    """Return all 2^bits discrete phase values in [0, 2π)."""
    levels = 2 ** bits
    return np.linspace(0, 2 * math.pi, levels, endpoint=False)


# ── Channel model (simplified) ────────────────────────────────────────────

def ris_effective_gain(phases: np.ndarray,
                       n_elements: int,
                       freq_thz: float = 0.3,
                       distance_m: float = 100.0) -> float:
    """
    Simplified RIS effective channel gain.
    |h_eff|² = |Σ_n exp(jφ_n) · a_n|²
    where a_n = element steering coefficients (uniform linear array model).
    """
    wavelength = SPEED_OF_LIGHT / (freq_thz * 1e12)
    d_element  = wavelength / 2          # half-wavelength spacing

    # Steering vector (broadside array, far-field approximation)
    n_idx = np.arange(n_elements)
    steering = np.exp(1j * math.pi * n_idx)   # θ=0° (broadside)

    # RIS reflection matrix Φ = diag(e^{jφ})
    reflection = np.exp(1j * phases[:n_elements])

    # Effective array response
    effective = np.dot(reflection, steering)
    gain_linear = abs(effective) ** 2

    # Convert to dB
    return 10 * math.log10(max(gain_linear, 1e-20))


def estimate_snr(phases: np.ndarray,
                 n_elements: int,
                 tx_power_dbm: float,
                 freq_thz: float,
                 noise_dbm: float = -90.0) -> float:
    """Estimate received SNR in dB given current phase configuration."""
    ris_gain_db = ris_effective_gain(phases, n_elements, freq_thz)
    # Simplified link budget
    path_loss   = 20 * math.log10(4 * math.pi * 100 * freq_thz * 1e12 / SPEED_OF_LIGHT)
    rx_dbm      = tx_power_dbm - path_loss + ris_gain_db
    snr_db      = rx_dbm - noise_dbm
    return snr_db


# ── Optimization algorithms ───────────────────────────────────────────────

def random_init(n_elements: int) -> np.ndarray:
    """Random phase initialisation in [0, 2π)."""
    return np.random.uniform(0, 2 * math.pi, n_elements)


def greedy_phase_search(n_elements: int,
                        phase_bits: int,
                        tx_power_dbm: float,
                        freq_thz: float,
                        n_iter: int = 3) -> np.ndarray:
    """
    Greedy element-wise phase search.
    For each RIS element, sweep all 2^bits phase levels and keep the best.
    Repeat for n_iter passes.
    """
    codebook = phase_to_codebook(phase_bits)
    phases   = random_init(n_elements)

    for _ in range(n_iter):
        for i in range(n_elements):
            best_snr   = -999.0
            best_phase = phases[i]
            for phi in codebook:
                phases[i] = phi
                snr = estimate_snr(phases, n_elements, tx_power_dbm, freq_thz)
                if snr > best_snr:
                    best_snr   = snr
                    best_phase = phi
            phases[i] = best_phase

    return quantize_phases(phases, phase_bits)


def hill_climbing(n_elements: int,
                  phase_bits: int,
                  tx_power_dbm: float,
                  freq_thz: float,
                  n_restarts: int = 5,
                  n_steps: int = 200) -> np.ndarray:
    """
    Random-restart hill climbing with Gaussian perturbation.
    Returns best phase vector found across all restarts.
    """
    best_phases = None
    best_snr    = -999.0
    step_size   = 2 * math.pi / (2 ** phase_bits)

    for _ in range(n_restarts):
        phases = random_init(n_elements)
        cur_snr = estimate_snr(phases, n_elements, tx_power_dbm, freq_thz)

        for _ in range(n_steps):
            candidate = phases + np.random.normal(0, step_size, n_elements)
            candidate = candidate % (2 * math.pi)
            snr_cand  = estimate_snr(candidate, n_elements, tx_power_dbm, freq_thz)
            if snr_cand > cur_snr:
                phases  = candidate
                cur_snr = snr_cand

        if cur_snr > best_snr:
            best_snr    = cur_snr
            best_phases = phases.copy()

    return quantize_phases(best_phases, phase_bits)


# ── Public API ────────────────────────────────────────────────────────────

class RISOptimizer:
    """
    High-level RIS optimizer.
    Strategy auto-selected based on element count:
      ≤ 64  → greedy (exact, fast)
      > 64  → hill climbing (approximate, scalable)
    """

    def optimize(self,
                 ris_elements: int,
                 target_snr_db: float,
                 phase_bits: int,
                 tx_power_dbm: float = 30.0,
                 freq_thz: float = 0.3) -> Dict[str, Any]:
        """
        Run optimization and return result dict.
        """
        strategy = "greedy" if ris_elements <= 64 else "hill_climbing"

        if strategy == "greedy":
            opt_phases = greedy_phase_search(
                ris_elements, phase_bits, tx_power_dbm, freq_thz)
        else:
            opt_phases = hill_climbing(
                ris_elements, phase_bits, tx_power_dbm, freq_thz)

        # Baseline (random phases)
        baseline_phases = random_init(ris_elements)
        baseline_snr    = estimate_snr(baseline_phases, ris_elements,
                                       tx_power_dbm, freq_thz)
        optimized_snr   = estimate_snr(opt_phases, ris_elements,
                                       tx_power_dbm, freq_thz)
        snr_gain        = optimized_snr - baseline_snr

        return {
            "ris_elements":        ris_elements,
            "optimized_phases":    [round(float(p), 4) for p in opt_phases],
            "estimated_snr_gain":  round(snr_gain, 3),
            "phase_bits":          phase_bits,
            "strategy":            strategy,
            "optimized_snr_db":    round(optimized_snr, 3),
            "baseline_snr_db":     round(baseline_snr, 3),
        }


# ── CLI test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    opt    = RISOptimizer()
    result = opt.optimize(
        ris_elements=64, target_snr_db=20.0,
        phase_bits=4, tx_power_dbm=30.0, freq_thz=0.3
    )
    print("Optimization result:")
    for k, v in result.items():
        if k != "optimized_phases":
            print(f"  {k}: {v}")
    print(f"  phases[:5]: {result['optimized_phases'][:5]} ...")

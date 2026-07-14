"""
predictor.py
AI-Based SNR Prediction — scikit-learn Random Forest
Trains on synthetic simulation data; saves/loads model with joblib.
"""

import os
import math
import random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple, Dict, Any

MODEL_PATH  = "snr_predictor.pkl"
SPEED_OF_LIGHT = 3e8


# ── Feature engineering ───────────────────────────────────────────────────

def build_features(velocity: float,
                   ris_elements: int,
                   tx_power_dbm: float,
                   frequency_thz: float,
                   num_users: int) -> np.ndarray:
    """
    Derives physics-informed features for the ML model.
    Features (8 total):
      1. velocity (km/h)
      2. ris_elements
      3. tx_power_dbm
      4. frequency_thz
      5. num_users
      6. ris_gain_db = 20·log10(N_RIS)      — array gain proxy
      7. doppler_hz = (v/3.6)·f_c/c         — Doppler shift
      8. path_loss_proxy = 20·log10(f_THz)  — frequency-dependent loss
    """
    ris_gain      = 20 * math.log10(max(ris_elements, 1))
    doppler       = (velocity / 3.6) * (frequency_thz * 1e12) / SPEED_OF_LIGHT
    path_loss_prx = 20 * math.log10(frequency_thz + 1e-6)
    return np.array([[velocity, ris_elements, tx_power_dbm, frequency_thz,
                      num_users, ris_gain, doppler, path_loss_prx]])


# ── Synthetic training data ───────────────────────────────────────────────

def _generate_training_data(n: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) pairs using physics simulation for training."""
    from simulation import run_simulation

    X, y = [], []
    for _ in range(n):
        vel   = random.uniform(0, 120)
        ris   = random.choice([16, 32, 64, 96, 128, 192, 256])
        pwr   = random.uniform(10, 46)
        freq  = random.uniform(0.1, 1.0)
        users = random.randint(1, 8)
        alpha = random.uniform(0, 1)

        result = run_simulation(vel, ris, pwr, alpha, freq, users, add_noise=True)
        feat   = build_features(vel, ris, pwr, freq, users)
        X.append(feat[0])
        y.append(result.snr_db)

    return np.array(X), np.array(y)


# ── Model training ────────────────────────────────────────────────────────

def train_model(n_samples: int = 3000, save: bool = True) -> Dict[str, Any]:
    """
    Train a Random Forest SNR predictor.
    Returns training metrics dict.
    """
    print(f"[Predictor] Generating {n_samples} training samples...")
    X, y = _generate_training_data(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("[Predictor] Training Random Forest...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    metrics = {
        "mae_db":       round(mae, 4),
        "r2_score":     round(r2, 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "model":        "RandomForestRegressor",
    }
    print(f"[Predictor] MAE={mae:.3f} dB | R²={r2:.4f}")

    if save:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"[Predictor] Model saved → {MODEL_PATH}")

    return metrics


# ── Inference ─────────────────────────────────────────────────────────────

class SNRPredictor:
    """Singleton wrapper — loads model once, serves predictions."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
        return cls._instance

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        if os.path.exists(MODEL_PATH):
            self._pipeline = joblib.load(MODEL_PATH)
            print("[Predictor] Model loaded from disk.")
        else:
            print("[Predictor] No saved model found — training now...")
            train_model(n_samples=2000)
            self._pipeline = joblib.load(MODEL_PATH)

    def predict(self,
                velocity: float,
                ris_elements: int,
                tx_power_dbm: float,
                frequency_thz: float,
                num_users: int) -> Dict[str, Any]:
        """
        Predict SNR in dB for given radio parameters.
        Returns dict with predicted_snr_db, confidence, model name.
        """
        self._ensure_loaded()
        X = build_features(velocity, ris_elements, tx_power_dbm,
                           frequency_thz, num_users)
        snr_pred = float(self._pipeline.predict(X)[0])

        # Confidence proxy: inter-tree std deviation (RF only)
        rf = self._pipeline.named_steps["rf"]
        scaler = self._pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        tree_preds = np.array([t.predict(X_scaled) for t in rf.estimators_])
        std_dev = float(tree_preds.std())
        confidence = max(0.0, min(1.0, 1.0 - std_dev / 20.0))

        return {
            "predicted_snr_db": round(snr_pred, 3),
            "confidence":       round(confidence, 3),
            "model":            "RandomForestRegressor",
        }

    def retrain(self, n_samples: int = 3000) -> Dict[str, Any]:
        """Retrain and hot-reload model."""
        metrics = train_model(n_samples=n_samples, save=True)
        self._pipeline = joblib.load(MODEL_PATH)
        return metrics


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = train_model(n_samples=3000)
    print("Training complete:", metrics)

    predictor = SNRPredictor()
    result = predictor.predict(
        velocity=45.0, ris_elements=64,
        tx_power_dbm=30.0, frequency_thz=0.3, num_users=4
    )
    print("Test prediction:", result)

"""
classifier.py
-------------
Wet/Dry cough classifier and multi-label disease prediction (Asthma, COPD, Healthy).
Uses RandomForest; supports training and inference with joblib-saved models.
"""

import os
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WET_DRY_MODEL_PATH = MODELS_DIR / "wet_dry_model.pkl"
DISEASE_MODEL_PATH = MODELS_DIR / "disease_model.pkl"


# --------------- Wet / Dry classification ---------------

def _wet_dry_features_from_cough_dict(f: dict) -> np.ndarray:
    """Build feature vector for one cough from feature_extraction output."""
    return np.concatenate([
        f["mfcc_mean"],
        [f["rms"], f["zcr"], f["centroid"], f["entropy"]],
    ])


def train_wet_dry_classifier(
    cough_features_list: list[dict],
    labels: list[int],
    save_path: str | Path = WET_DRY_MODEL_PATH,
) -> RandomForestClassifier:
    """
    Train RandomForest for wet (1) vs dry (0).
    cough_features_list: list of dicts from feature_extraction.features_per_cough.
    labels: 0 = dry, 1 = wet.
    """
    X = np.array([_wet_dry_features_from_cough_dict(f) for f in cough_features_list])
    y = np.array(labels, dtype=int)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    return clf


def load_wet_dry_model(path: str | Path = WET_DRY_MODEL_PATH):
    """Load saved wet/dry classifier."""
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_wet_dry(
    cough_features_list: list[dict],
    model_path: str | Path = WET_DRY_MODEL_PATH,
) -> list[int]:
    """
    Predict 0 (dry) or 1 (wet) for each cough.
    If model file missing, use heuristic: higher RMS/centroid -> more likely wet.
    """
    clf = load_wet_dry_model(model_path)
    if clf is not None:
        X = np.array([_wet_dry_features_from_cough_dict(f) for f in cough_features_list])
        return clf.predict(X).tolist()
    # Heuristic fallback: wet ~ higher RMS and often higher centroid
    out = []
    for f in cough_features_list:
        rms, cent = f["rms"], f["centroid"]
        # Simple rule: above median-like thresholds -> wet
        wet_score = (rms > 0.02) + (cent > 1500)
        out.append(1 if wet_score >= 1 else 0)
    return out


# --------------- Disease prediction (Asthma, COPD, Healthy) ---------------

# Risk level buckets
def probability_to_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Medium"
    return "High"


def train_disease_classifier(
    X: np.ndarray,
    y: list[str],
    save_path: str | Path = DISEASE_MODEL_PATH,
) -> tuple[RandomForestClassifier, LabelEncoder]:
    """
    Train multi-class classifier for Asthma / COPD / Healthy.
    y: list of labels ("asthma" | "copd" | "healthy").
    Saves model and label encoder (for class names).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
    clf.fit(X, y_enc)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, save_path)
    return clf, le


def load_disease_model(path: str | Path = DISEASE_MODEL_PATH):
    """Load saved disease model and label encoder."""
    path = Path(path)
    if not path.exists():
        return None, None
    data = joblib.load(path)
    return data["model"], data.get("label_encoder")


def predict_disease_proba(
    feature_vector: np.ndarray,
    model_path: str | Path = DISEASE_MODEL_PATH,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Predict class probabilities. feature_vector is 1D (fusion vector).
    Returns (dict with keys asthma, copd, healthy -> probability), proba_array.
    If model missing, returns uniform probabilities.
    """
    clf, le = load_disease_model(model_path)
    if clf is None or le is None:
        return (
            {"asthma": 1.0 / 3, "copd": 1.0 / 3, "healthy": 1.0 / 3},
            np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        )
    # Ensure 2D
    X = np.atleast_2d(feature_vector)
    if X.shape[1] != clf.n_features_in_:
        # Pad or truncate to match training
        n = clf.n_features_in_
        if X.shape[1] < n:
            X = np.pad(X, ((0, 0), (0, n - X.shape[1])), constant_values=0)
        else:
            X = X[:, :n]
    proba = clf.predict_proba(X)[0]
    classes = le.classes_
    out = {}
    for i, c in enumerate(classes):
        name = c.lower() if hasattr(c, "lower") else str(c).lower()
        out[name] = float(proba[i])
    # Ensure keys exist
    for k in ["asthma", "copd", "healthy"]:
        if k not in out:
            out[k] = 0.0
    return out, proba


def get_asthma_copd_risk_levels(proba_dict: dict) -> dict[str, str]:
    """Map asthma and COPD probabilities to Low/Medium/High."""
    return {
        "asthma": probability_to_risk_level(proba_dict.get("asthma", 0.0)),
        "copd": probability_to_risk_level(proba_dict.get("copd", 0.0)),
    }

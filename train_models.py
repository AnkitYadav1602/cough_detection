"""
train_models.py
---------------
Script to train and save wet/dry classifier and disease (Asthma/COPD/Healthy) model.
Run this when you have labeled data. Without data, creates placeholder models
using synthetic data so the pipeline runs end-to-end.
"""

import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WET_DRY_PATH = MODELS_DIR / "wet_dry_model.pkl"
DISEASE_PATH = MODELS_DIR / "disease_model.pkl"

# Feature dimensions
WET_DRY_FEAT_DIM = 13 + 4  # 13 MFCC + rms, zcr, centroid, entropy
FUSION_DIM = 10 + 5 + 13   # questionnaire 10 + audio agg 5 + mfcc 13 = 28


def train_wet_dry_with_synthetic():
    """
    Train wet/dry classifier on synthetic data so the app runs without real labels.
    In production, replace with real (cough_features_list, labels).
    """
    np.random.seed(42)
    n_samples = 200
    # Dry: lower RMS, lower centroid on average. Wet: higher.
    X = np.random.randn(n_samples, WET_DRY_FEAT_DIM) * 2
    # Simulate: dry (0) has lower energy (col 13) and centroid (col 15)
    labels = (X[:, 13] + X[:, 15] / 3000 > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, labels)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, WET_DRY_PATH)
    print(f"Saved wet/dry model to {WET_DRY_PATH}")
    return clf


def train_disease_with_synthetic():
    """
    Train Asthma/COPD/Healthy classifier on synthetic fusion vectors.
    In production, replace with real (X_fusion, y_labels).
    """
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, FUSION_DIM) * 1.5
    # Simple rule for synthetic labels
    asthma_score = X[:, 2] + X[:, 3] + X[:, 7]   # wheezing, mucus, family_asthma
    copd_score = X[:, 1] + X[:, 9] + X[:, 12]    # smoking, smoke_exposure, cough_count
    healthy = (asthma_score < 0) & (copd_score < 0)
    asthma = (asthma_score >= 0.5) & (asthma_score >= copd_score)
    copd = (copd_score >= 0.5) & (copd_score > asthma_score)
    y = np.where(healthy, "healthy", np.where(asthma, "asthma", "copd"))
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
    clf.fit(X, y_enc)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, DISEASE_PATH)
    print(f"Saved disease model to {DISEASE_PATH}")
    return clf, le


def train_with_real_data_wet_dry(cough_features_list: list, labels: list):
    """
    Use your own data: list of feature dicts from feature_extraction.features_per_cough,
    and labels 0 (dry) / 1 (wet).
    """
    from classifier import _wet_dry_features_from_cough_dict, train_wet_dry_classifier
    train_wet_dry_classifier(cough_features_list, labels, save_path=WET_DRY_PATH)


def train_with_real_data_disease(X: np.ndarray, y: list):
    """
    Use your own data: X = fusion vectors (n_samples, FUSION_DIM), y = list of
    "asthma" | "copd" | "healthy".
    """
    from classifier import train_disease_classifier
    train_disease_classifier(X, y, save_path=DISEASE_PATH)


if __name__ == "__main__":
    print("Training wet/dry classifier (synthetic data)...")
    train_wet_dry_with_synthetic()
    print("Training disease classifier (synthetic data)...")
    train_disease_with_synthetic()
    print("Done. Models saved under models/")

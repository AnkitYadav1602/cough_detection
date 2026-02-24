"""
fusion.py
---------
Feature fusion: combine questionnaire (numeric encoding) with aggregate
cough/acoustic features into a single vector for disease prediction.
"""

import numpy as np


# Questionnaire keys and encoding
QUESTIONNAIRE_KEYS = [
    "age",
    "gender_encoded",
    "smoking",
    "wheezing",
    "mucus",
    "night_cough",
    "frequent_cough",
    "difficulty_breathing",
    "family_asthma",
    "smoke_exposure",
]

AUDIO_AGG_KEYS = [
    "mean_energy",
    "mean_zcr",
    "mean_centroid",
    "wet_percentage",
    "cough_count",
]

MFCC_KEYS = [f"mfcc_{i}_mean" for i in range(1, 14)]


def encode_questionnaire(q: dict) -> dict:
    """
    Convert questionnaire JSON to numeric encoding.
    - Yes/No, true/false -> 1/0
    - gender: male=1, female=0, other=0.5 (or map as needed)
    - age: keep numeric
    """
    out = {}
    # Age
    out["age"] = float(q.get("age", 30))
    # Gender: male=1, female=0, other=0.5
    g = (q.get("gender") or "").lower().strip()
    if g == "male":
        out["gender_encoded"] = 1.0
    elif g == "female":
        out["gender_encoded"] = 0.0
    else:
        out["gender_encoded"] = 0.5
    # Booleans
    for key in [
        "smoking",
        "wheezing",
        "mucus",
        "night_cough",
        "frequent_cough",
        "difficulty_breathing",
        "family_asthma",
        "smoke_exposure",
    ]:
        v = q.get(key, False)
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, str):
            out[key] = 1.0 if v.lower() in ("yes", "true", "1") else 0.0
        else:
            out[key] = 1.0 if v else 0.0
    return out


def build_fusion_vector(
    questionnaire_encoded: dict,
    aggregate_cough_features: dict,
) -> np.ndarray:
    """
    Build unified feature vector in fixed order:
    [age, gender_encoded, smoking, wheezing, mucus, night_cough, frequent_cough,
     difficulty_breathing, family_asthma, smoke_exposure,
     mean_energy, mean_zcr, mean_centroid, wet_percentage, cough_count,
     mfcc_1_mean, ..., mfcc_13_mean]
    """
    parts = []
    for k in QUESTIONNAIRE_KEYS:
        parts.append(questionnaire_encoded.get(k, 0.0))
    for k in AUDIO_AGG_KEYS:
        parts.append(aggregate_cough_features.get(k, 0.0))
    for k in MFCC_KEYS:
        parts.append(aggregate_cough_features.get(k, 0.0))
    return np.array(parts, dtype=np.float64)


def fuse(
    questionnaire: dict,
    aggregate_cough_features: dict,
) -> np.ndarray:
    """
    One-shot: encode questionnaire, then build fusion vector.
    """
    enc = encode_questionnaire(questionnaire)
    return build_fusion_vector(enc, aggregate_cough_features)

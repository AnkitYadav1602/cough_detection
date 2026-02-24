"""
coughvid_loader.py
------------------
Load COUGHVID dataset: metadata CSV + audio files.
Supports Zenodo/Kaggle layout and multiple audio formats (wav, webm, ogg).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import librosa

from feature_extraction import features_per_cough, aggregate_cough_features
from fusion import build_fusion_vector

# Default COUGHVID layout: data_dir contains metadata CSV and a folder of audio files
DEFAULT_SR = 22050

# Common CSV column names across COUGHVID v2/v3
COL_UUID = "uuid"
COL_COUGH_TYPE = "cough_type"
COL_AGE = "age"
COL_GENDER = "gender"
COL_RESPIRATORY = "respiratory_condition"
COL_STATUS = "status"
COL_STATUS_SSL = "status_SSL"

# Cough type values in metadata
COUGH_DRY = ("dry", "0", "dry_cough")
COUGH_WET = ("wet", "1", "wet_cough")


def find_audio_path(data_dir: Path, uuid_or_name: str, extensions: tuple = (".wav", ".webm", ".ogg", ".mp3")) -> Path | None:
    """Locate audio file by uuid or base name in data_dir, data_dir/recordings, or parent/recordings."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{uuid_or_name}{ext}"
        for ext in extensions
    ]
    for p in candidates:
        if p.exists():
            return p
    # Try recordings subfolder (Kaggle layout)
    for ext in extensions:
        p = data_dir / "recordings" / f"{uuid_or_name}{ext}"
        if p.exists():
            return p
    # Try parent/recordings (e.g. data/public_dataset_v3/coughvid_20211012 -> parent/recordings)
    for ext in extensions:
        p = data_dir.parent / "recordings" / f"{uuid_or_name}{ext}"
        if p.exists():
            return p
    return None


def load_coughvid_metadata(
    data_dir: str | Path,
    metadata_file: str = "metadata_compiled.csv",
) -> pd.DataFrame:
    """Load COUGHVID metadata CSV; try common filenames if not found."""
    data_dir = Path(data_dir)
    for name in (metadata_file, "metadata_compiled.csv", "coughvid_metadata.csv", "metadata.csv"):
        p = data_dir / name
        if p.exists():
            df = pd.read_csv(p)
            # Normalize column names: lowercase and strip
            df.columns = [c.strip().lower() if isinstance(c, str) else c for c in df.columns]
            return df
    raise FileNotFoundError(f"No metadata CSV found in {data_dir}. Tried: {metadata_file}, metadata_compiled.csv, metadata.csv")


def normalize_cough_type(value) -> int | None:
    """Map cough_type to 0 (dry) or 1 (wet). Returns None if unknown."""
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in ("dry", "0", "dry_cough"):
        return 0
    if v in ("wet", "1", "wet_cough", "productive"):
        return 1
    return None


def _cough_type_from_row(row: pd.Series) -> int | None:
    """Get cough type from row, trying cough_type then cough_type_1..4 (first non-null)."""
    for col in ("cough_type", "cough_type_1", "cough_type_2", "cough_type_3", "cough_type_4"):
        if col in row.index:
            label = normalize_cough_type(row.get(col))
            if label is not None:
                return label
    return None


def normalize_disease_label(
    row: pd.Series,
    respiratory_col: str = COL_RESPIRATORY,
    status_col: str = COL_STATUS,
    status_ssl_col: str = COL_STATUS_SSL,
) -> str | None:
    """
    Map COUGHVID columns to our classes: healthy, asthma, copd.
    - respiratory_condition present / positive -> asthma
    - COVID-19 status positive -> copd (as proxy for respiratory disease)
    - Otherwise -> healthy
    """
    resp = row.get(respiratory_col) if respiratory_col in row.index else None
    status = row.get(status_col) if status_col in row.index else None
    ssl = row.get(status_ssl_col) if status_ssl_col in row.index else None
    # Prefer SSL if available (V3)
    val = ssl if pd.notna(ssl) and str(ssl).strip() else status
    if pd.notna(val):
        v = str(val).strip().lower()
        if v in ("positive", "covid", "covid-19", "1", "yes"):
            return "copd"
        if v in ("negative", "healthy", "0", "no"):
            return "healthy"
        if v in ("symptomatic",):
            return "asthma"
    if pd.notna(resp) and str(resp).strip().lower() in ("1", "yes", "true", "positive"):
        return "asthma"
    return "healthy"


def load_audio_safe(path: Path, sr: int = DEFAULT_SR) -> np.ndarray | None:
    """Load audio file; return None on error."""
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception:
        return None


def prepare_wet_dry_data(
    data_dir: Path,
    df: pd.DataFrame,
    uuid_col: str | None = None,
    cough_type_col: str | None = None,
    max_samples: int | None = 5000,
    sr: int = DEFAULT_SR,
):
    """
    Load audio and extract per-cough features for rows with valid cough_type.
    Returns (list of feature dicts, list of labels 0/1).
    If uuid_col/cough_type_col are None, they are auto-detected from CSV columns.
    """
    data_dir = Path(data_dir)
    # Auto-detect uuid column
    if uuid_col is None:
        for c in (COL_UUID, "id", "filename", "file_name"):
            if c in df.columns:
                uuid_col = c
                break
        if uuid_col is None:
            uuid_col = df.columns[0]
    # Use row-based cough type (supports cough_type_1..4)
    feature_list = []
    labels_list = []
    count = 0
    for _, row in df.iterrows():
        if max_samples and count >= max_samples:
            break
        label = _cough_type_from_row(row)
        if label is None:
            continue
        uid = row.get(uuid_col)
        if pd.isna(uid):
            continue
        path = find_audio_path(data_dir, str(uid).strip())
        if path is None:
            continue
        y = load_audio_safe(path, sr=sr)
        if y is None or len(y) < 256:
            continue
        feats = features_per_cough(y, sr=sr)
        feature_list.append(feats)
        labels_list.append(label)
        count += 1
    return feature_list, labels_list


def prepare_disease_data(
    data_dir: Path,
    df: pd.DataFrame,
    uuid_col: str | None = None,
    questionnaire_cols: dict | None = None,
    max_samples: int | None = 3000,
    sr: int = DEFAULT_SR,
):
    """
    For each row with valid audio and disease label, build fusion vector and label.
    questionnaire_cols: optional map from our key to CSV column name (e.g. {"age": "age", "gender": "gender"}).
    Returns (X fusion matrix, y list of "healthy"|"asthma"|"copd").
    """
    data_dir = Path(data_dir)
    if uuid_col is None:
        for c in (COL_UUID, "id", "filename"):
            if c in df.columns:
                uuid_col = c
                break
        if uuid_col is None:
            uuid_col = df.columns[0]
    q_cols = questionnaire_cols or {}
    X_list = []
    y_list = []
    count = 0
    for _, row in df.iterrows():
        if max_samples and count >= max_samples:
            break
        disease = normalize_disease_label(row)
        if disease is None:
            continue
        uid = row.get(uuid_col)
        if pd.isna(uid):
            continue
        path = find_audio_path(data_dir, str(uid).strip())
        if path is None:
            continue
        y_audio = load_audio_safe(path, sr=sr)
        if y_audio is None or len(y_audio) < 256:
            continue
        # One recording = one cough for COUGHVID
        feats = features_per_cough(y_audio, sr=sr)
        wet_label = normalize_cough_type(row.get(COL_COUGH_TYPE))
        if wet_label is None:
            wet_label = 0
        agg = aggregate_cough_features([y_audio], [wet_label], sr=sr)
        # Questionnaire from metadata
        age_val = row.get(q_cols.get("age", COL_AGE), 30)
        try:
            age = float(age_val) if pd.notna(age_val) else 30.0
        except (TypeError, ValueError):
            age = 30.0
        g = row.get(q_cols.get("gender", COL_GENDER), "unknown")
        if pd.isna(g):
            g = "unknown"
        g = str(g).strip().lower()
        gender_enc = 1.0 if g == "male" else (0.0 if g == "female" else 0.5)
        q_enc = {
            "age": age,
            "gender_encoded": gender_enc,
            "smoking": 0.0,
            "wheezing": 0.0,
            "mucus": 0.0,
            "night_cough": 0.0,
            "frequent_cough": 0.0,
            "difficulty_breathing": 0.0,
            "family_asthma": 0.0,
            "smoke_exposure": 0.0,
        }
        for k in q_enc:
            if k in ("age", "gender_encoded"):
                continue
            col = q_cols.get(k)
            if col and col in row.index:
                v = row[col]
                q_enc[k] = 1.0 if (pd.notna(v) and str(v).strip().lower() in ("1", "yes", "true")) else 0.0
        vec = build_fusion_vector(q_enc, agg)
        X_list.append(vec)
        y_list.append(disease)
        count += 1
    if not X_list:
        return np.zeros((0, 28)), []
    return np.array(X_list), y_list

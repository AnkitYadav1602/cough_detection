"""
train_on_coughvid.py
--------------------
Train wet/dry and disease models using the COUGHVID dataset.
Download COUGHVID from Kaggle or Zenodo and place it in data/coughvid/ (see README in data/).
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "coughvid"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Train models on COUGHVID dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to COUGHVID folder (contains metadata CSV and audio files)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata_compiled.csv",
        help="Metadata CSV filename",
    )
    parser.add_argument(
        "--max-wet-dry",
        type=int,
        default=5000,
        help="Max samples for wet/dry training (default 5000)",
    )
    parser.add_argument(
        "--max-disease",
        type=int,
        default=3000,
        help="Max samples for disease training (default 3000)",
    )
    parser.add_argument(
        "--skip-wet-dry",
        action="store_true",
        help="Skip training wet/dry model",
    )
    parser.add_argument(
        "--skip-disease",
        action="store_true",
        help="Skip training disease model",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Download COUGHVID from:")
        print("  Kaggle: https://www.kaggle.com/datasets/orvile/coughvid-v3")
        print("  Zenodo:  https://zenodo.org/records/7024894")
        print(f"Then extract so that '{data_dir}' contains the metadata CSV and audio (e.g. recordings/).")
        return 1

    from coughvid_loader import (
        load_coughvid_metadata,
        prepare_wet_dry_data,
        prepare_disease_data,
    )
    from classifier import train_wet_dry_classifier, train_disease_classifier

    print("Loading COUGHVID metadata...")
    try:
        df = load_coughvid_metadata(data_dir, metadata_file=args.metadata)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f"  Rows: {len(df)}, columns: {list(df.columns)[:15]}...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_wet_dry:
        print("Preparing wet/dry data (extracting features from audio)...")
        feat_list, labels = prepare_wet_dry_data(
            data_dir,
            df,
            max_samples=args.max_wet_dry,
        )
        if len(feat_list) < 20:
            print(f"  Too few samples with valid cough_type ({len(feat_list)}). Need at least 20.")
            print("  Ensure metadata has 'cough_type' column with values 'dry' or 'wet'.")
        else:
            print(f"  Samples: {len(feat_list)} (dry={sum(1 for l in labels if l==0)}, wet={sum(1 for l in labels if l==1)})")
            train_wet_dry_classifier(feat_list, labels, save_path=MODELS_DIR / "wet_dry_model.pkl")
            print("  Saved wet_dry_model.pkl")
        print()

    if not args.skip_disease:
        print("Preparing disease data (fusion vectors)...")
        X, y = prepare_disease_data(
            data_dir,
            df,
            max_samples=args.max_disease,
        )
        if X.shape[0] < 30:
            print(f"  Too few samples ({X.shape[0]}). Need at least 30.")
        else:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Samples: {X.shape[0]}, classes: {dict(zip(unique.tolist(), counts.tolist()))}")
            train_disease_classifier(X, y, save_path=MODELS_DIR / "disease_model.pkl")
            print("  Saved disease_model.pkl")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())

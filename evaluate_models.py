"""
evaluate_models.py
------------------
Evaluate the wet/dry and disease classifiers on COUGHVID samples.
Prints accuracy / classification reports for quick sanity checks.
"""

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from coughvid_loader import (
    load_coughvid_metadata,
    prepare_wet_dry_data,
    prepare_disease_data,
)
from classifier import (
    load_wet_dry_model,
    load_disease_model,
    _wet_dry_features_from_cough_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "public_dataset_v3" / "coughvid_20211012"


def eval_wet_dry(data_dir: Path, metadata: str, max_samples: int):
    df = load_coughvid_metadata(data_dir, metadata_file=metadata)
    feature_list, labels = prepare_wet_dry_data(data_dir, df, max_samples=max_samples)
    if not feature_list:
        print("No wet/dry samples found.")
        return
    model = load_wet_dry_model()
    if model is None:
        print("Wet/dry model not found.")
        return
    X = [ _wet_dry_features_from_cough_dict(f) for f in feature_list ]
    preds = model.predict(X)
    print("\n=== Wet/Dry classifier ===")
    print(f"Samples: {len(labels)}")
    print("Accuracy:", accuracy_score(labels, preds))
    print("Classification report:")
    print(classification_report(labels, preds, labels=[0,1], digits=3))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))


def eval_disease(data_dir: Path, metadata: str, max_samples: int):
    df = load_coughvid_metadata(data_dir, metadata_file=metadata)
    X, y = prepare_disease_data(data_dir, df, max_samples=max_samples)
    if X.shape[0] == 0:
        print("No disease samples found.")
        return
    model, le = load_disease_model()
    if model is None or le is None:
        print("Disease model not found.")
        return
    preds_enc = model.predict(X)
    preds = le.inverse_transform(preds_enc)
    print("\n=== Disease classifier ===")
    print(f"Samples: {len(y)}, classes: {sorted(set(y))}")
    print("Accuracy:", accuracy_score(y, preds))
    print("Classification report:")
    print(classification_report(y, preds, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y, preds))


def main():
    parser = argparse.ArgumentParser(description="Evaluate COUGHVID classifiers")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata", type=str, default="metadata_compiled.csv")
    parser.add_argument("--max-wet-dry", type=int, default=2000)
    parser.add_argument("--max-disease", type=int, default=1500)
    parser.add_argument("--skip-wet-dry", action="store_true")
    parser.add_argument("--skip-disease", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    if not args.skip_wet_dry:
        eval_wet_dry(args.data_dir, args.metadata, args.max_wet_dry)
    if not args.skip_disease:
        eval_disease(args.data_dir, args.metadata, args.max_disease)


if __name__ == "__main__":
    main()

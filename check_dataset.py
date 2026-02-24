"""
check_dataset.py
----------------
Verify COUGHVID dataset layout: metadata CSV, columns, and audio file locations.
Run from project root:  uv run python check_dataset.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def find_audio_path(data_dir: Path, uuid_or_name: str, extensions: tuple = (".wav", ".webm", ".ogg", ".mp3")) -> Path | None:
    data_dir = Path(data_dir)
    for ext in extensions:
        p = data_dir / f"{uuid_or_name}{ext}"
        if p.exists():
            return p
    for ext in extensions:
        p = data_dir / "recordings" / f"{uuid_or_name}{ext}"
        if p.exists():
            return p
    for ext in extensions:
        p = data_dir.parent / "recordings" / f"{uuid_or_name}{ext}"
        if p.exists():
            return p
    return None


def main():
    print("Dataset check for cough_ai")
    print("=" * 50)
    if not DATA_ROOT.exists():
        print(f"Data folder not found: {DATA_ROOT}")
        return
    # Find all metadata CSVs
    csvs = list(DATA_ROOT.rglob("*.csv"))
    if not csvs:
        print("No CSV files found under data/")
        return
    print(f"Found {len(csvs)} CSV(s):")
    for p in sorted(csvs):
        print(f"  - {p.relative_to(PROJECT_ROOT)}")
    # Prefer metadata_compiled in public_dataset_v3
    meta_path = DATA_ROOT / "public_dataset_v3" / "coughvid_20211012" / "metadata_compiled.csv"
    if not meta_path.exists():
        meta_path = next((p for p in csvs if "metadata_compiled" in p.name or "coughvid_v3" in p.name), csvs[0])
    else:
        meta_path = Path(meta_path)
    print(f"\nUsing metadata: {meta_path.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(meta_path)
    df.columns = [c.strip().lower() if isinstance(c, str) else c for c in df.columns]
    print(f"  Rows: {len(df)}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)[:20]}...")
    # UUID column
    uuid_col = None
    for c in ("uuid", "id", "file_name", "audio_name"):
        if c in df.columns:
            uuid_col = c
            break
    if uuid_col is None:
        uuid_col = df.columns[0]
    print(f"  UUID/file column: '{uuid_col}'")
    # Cough type columns
    ct_cols = [c for c in df.columns if "cough_type" in c]
    print(f"  Cough type column(s): {ct_cols}")
    if ct_cols:
        sample = df[ct_cols[0]].dropna().head(5).tolist()
        print(f"    Sample values: {sample}")
    # Status
    for sc in ("status_ssl", "status"):
        if sc in df.columns:
            print(f"  {sc} sample: {df[sc].dropna().value_counts().head(5).to_dict()}")
            break
    # Audio check
    data_dir = meta_path.parent
    uuids = df[uuid_col].dropna().astype(str).str.strip()
    if "audio_name" in df.columns:
        uuids = uuids.str.replace(r"\.(wav|webm|ogg|json)$", "", case=False, regex=True)
    uuids = uuids.unique()[:10]
    print(f"\n  Checking audio for first 10 UUIDs (dir: {data_dir})...")
    found = 0
    for uid in uuids:
        path = find_audio_path(data_dir, uid)
        if path:
            found += 1
            print(f"    OK  {uid[:20]}... -> {path.name}")
        else:
            print(f"    MISSING  {uid[:20]}...")
    print(f"  Found {found}/10 audio files.")
    if found == 0:
        print("\n  Audio not found in this folder. Typical layouts:")
        print("    - metadata_compiled.csv and recordings/*.wav in the SAME folder")
        print("    - or recordings/ as subfolder of the folder containing the CSV")
        print("    - or recordings/ as sibling of the folder containing the CSV")
        print("  Ensure the zip contained audio files and they were extracted.")
    else:
        print("\n  Dataset layout looks good. Train with:")
        print(f"    uv run python train_on_coughvid.py --data-dir \"{data_dir}\"")


if __name__ == "__main__":
    main()

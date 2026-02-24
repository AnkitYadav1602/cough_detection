# Cough-Based Respiratory Screening System

End-to-end Python project for clinical-style respiratory screening from cough audio and a health questionnaire. The system detects cough events, classifies wet vs dry coughs, fuses acoustic and questionnaire features, and produces Asthma/COPD probability estimates plus a lung health index and a structured report.

## Requirements

- **Python 3.10+**
- Dependencies in `requirements.txt`: librosa, numpy, scikit-learn, joblib, streamlit, soundfile, scipy, pandas

## Project Structure

```
cough_ai/
├── data/
│   └── README.md             # COUGHVID download and layout
├── models/
│   ├── wet_dry_model.pkl      # Wet/dry cough classifier
│   ├── disease_model.pkl     # Asthma/COPD/Healthy classifier
├── audio_processing.py       # Cough detection and segmentation
├── feature_extraction.py     # MFCC, RMS, ZCR, spectral features
├── classifier.py             # Wet/dry + disease prediction
├── coughvid_loader.py        # COUGHVID metadata + audio loader
├── fusion.py                 # Questionnaire + audio feature fusion
├── report_generator.py       # Clinical report and lung health index
├── train_models.py           # Train with synthetic data
├── train_on_coughvid.py      # Train on COUGHVID dataset
├── app.py                    # Streamlit UI
├── requirements.txt
└── README.md
```

## Installation

**Option A: Using uv (recommended)**

```bash
cd cough_ai
uv venv
uv sync
python train_models.py
uv run streamlit run app.py
```

Or use the run script (from the `cough_ai` folder):

- **Windows (PowerShell):** `.\run.ps1`
- **Windows (CMD):** `run.bat`

**Option B: Using pip**

```bash
cd cough_ai
pip install -r requirements.txt
```

## Training the Models

### Option A: Train on COUGHVID (recommended)

1. Download COUGHVID from [Kaggle](https://www.kaggle.com/datasets/orvile/coughvid-v3) or [Zenodo](https://zenodo.org/records/7024894).
2. Extract so that `data/coughvid/` contains the metadata CSV and audio files (see `data/README.md`).
3. Run:

```bash
uv run python train_on_coughvid.py --data-dir data/coughvid
```

This trains both the wet/dry classifier and the disease (Asthma/COPD/Healthy) model from COUGHVID labels. Options: `--max-wet-dry 5000`, `--max-disease 3000`, `--metadata <filename>`, `--skip-wet-dry`, `--skip-disease`.

### Option B: Synthetic data (no dataset)

Models can be created without real data so the app runs end-to-end:

```bash
python train_models.py
```

- `models/wet_dry_model.pkl` – wet (1) vs dry (0) cough classifier  
- `models/disease_model.pkl` – Asthma / COPD / Healthy classifier  

### Training with Your Own Data

**Wet/Dry classifier**

- Extract per-cough features with `feature_extraction.features_per_cough()` for each cough segment.
- Build lists: `cough_features_list` (list of those dicts) and `labels` (0 = dry, 1 = wet).
- Call `train_with_real_data_wet_dry(cough_features_list, labels)` from `train_models.py`, or use `classifier.train_wet_dry_classifier()` directly.

**Disease classifier**

- Build fusion vectors with `fusion.fuse(questionnaire_dict, aggregate_cough_features)` for each subject.
- Collect `X` (array of fusion vectors) and `y` (list of labels: `"asthma"`, `"copd"`, `"healthy"`).
- Call `train_with_real_data_disease(X, y)` from `train_models.py`, or use `classifier.train_disease_classifier(X, y)` directly.

## Running the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (e.g. http://localhost:8501).

**In the UI:**

1. Fill in the **Health Questionnaire** (age, gender, symptoms, etc.).
2. **Upload a WAV file** containing one or more coughs from a single session.
3. Click **Analyze**.
4. View the **Screening Summary** (cough counts, lung health index, severity) and the full **Clinical Report**.

## System Flow

1. **User input** – WAV file + questionnaire (JSON-like form).
2. **Audio preprocessing** – Load and resample (e.g. 22.05 kHz).
3. **Cough detection** – Energy-based segmentation; consecutive high-energy frames grouped; segments &lt; 200 ms removed.
4. **Per-cough features** – MFCC (13), RMS, zero-crossing rate, spectral centroid, spectral entropy.
5. **Wet vs dry** – RandomForest (or heuristic if no model).
6. **Aggregate stats** – Cough count, wet/dry counts, wet %, mean features.
7. **Feature fusion** – Questionnaire (numeric) + aggregate audio features → single vector.
8. **Disease prediction** – Asthma / COPD / Healthy probabilities and risk levels (Low / Medium / High).
9. **Lung health index** – 0–100 from wet %, cough count, and COPD/Asthma probabilities.
10. **Report** – Structured clinical-style report.
11. **Streamlit** – Upload, form, Analyze, display report.

## Questionnaire JSON Example

```json
{
  "age": 26,
  "gender": "male",
  "smoking": false,
  "wheezing": false,
  "mucus": false,
  "night_cough": false,
  "frequent_cough": false,
  "difficulty_breathing": false,
  "family_asthma": false,
  "smoke_exposure": false
}
```

## Outputs

- **Total / wet / dry cough count** and **wetness %**
- **Lung health index** (0–100)
- **Asthma** and **COPD** probability and risk level (Low / Medium / High)
- **Overall abnormality severity** (Normal / Mild / Moderate / Severe)
- **Clinical-style report** with all of the above

## Notes

- No LLMs are used for feature extraction or core prediction; only deterministic/heuristic and ML models.
- An optional LLM could be used solely for final report wording if desired.
- For production, replace synthetic training in `train_models.py` with real labeled data and retrain both models.

"""
app.py
------
Streamlit UI for Cough-Based Respiratory Screening System.
Medical screening kiosk-style: WAV upload + questionnaire form -> Analyze -> Report.
"""

import io
import json
import streamlit as st
import numpy as np

from audio_processing import detect_cough_segments
from feature_extraction import features_per_cough, aggregate_cough_features
from classifier import (
    predict_wet_dry,
    predict_disease_proba,
    get_asthma_copd_risk_levels,
)
from fusion import fuse
from report_generator import (
    compute_lung_health_index,
    overall_severity,
    build_report_from_pipeline,
)

st.set_page_config(
    page_title="Cough-Based Respiratory Screening",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for medical kiosk look
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton>button {
        width: 100%;
        background-color: #1e5f74;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2d8ba8;
        color: white;
    }
    .report-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    h1 { color: #1e5f74; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e5f74;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def run_pipeline(wav_bytes: bytes, questionnaire: dict) -> dict | None:
    """
    Run full pipeline: audio -> coughs -> features -> wet/dry -> fusion -> disease -> report.
    Returns result dict or None on error.
    """
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=22050, mono=True)
        segments = detect_cough_segments(y, sr=sr)
        coughs = [y[s:e] for s, e in segments]

        if not coughs:
            return {
                "error": "No cough events detected in the audio. Please ensure the recording contains clear cough sounds.",
                "cough_count": 0,
            }

        # Per-cough features and wet/dry
        feats_list = [features_per_cough(c, sr=sr) for c in coughs]
        wet_dry = predict_wet_dry(feats_list)
        agg = aggregate_cough_features(coughs, wet_dry, sr=sr)

        # Fusion and disease prediction
        fusion_vec = fuse(questionnaire, agg)
        disease_proba, _ = predict_disease_proba(fusion_vec)
        risk_levels = get_asthma_copd_risk_levels(disease_proba)

        # Lung health index and severity
        lung_health = compute_lung_health_index(
            agg["wet_percentage"],
            agg["cough_count"],
            disease_proba.get("copd", 0.0),
            disease_proba.get("asthma", 0.0),
        )
        severity = overall_severity(
            lung_health,
            disease_proba.get("asthma", 0.0),
            disease_proba.get("copd", 0.0),
        )

        report_text = build_report_from_pipeline(
            questionnaire=questionnaire,
            cough_count=agg["cough_count"],
            wet_count=agg["wet_count"],
            dry_count=agg["dry_count"],
            wet_percentage=agg["wet_percentage"],
            disease_proba=disease_proba,
            risk_levels=risk_levels,
        )

        return {
            "error": None,
            "cough_count": agg["cough_count"],
            "wet_count": agg["wet_count"],
            "dry_count": agg["dry_count"],
            "wet_percentage": agg["wet_percentage"],
            "lung_health_index": lung_health,
            "asthma_probability": disease_proba.get("asthma", 0.0),
            "asthma_risk": risk_levels.get("asthma", "Low"),
            "copd_probability": disease_proba.get("copd", 0.0),
            "copd_risk": risk_levels.get("copd", "Low"),
            "healthy_probability": disease_proba.get("healthy", 0.0),
            "severity": severity,
            "report_text": report_text,
        }
    except Exception as e:
        return {"error": str(e), "cough_count": 0}


def main():
    st.title("🫁 Cough-Based Respiratory Screening System")
    st.markdown("Upload a WAV recording and complete the health questionnaire to receive a clinical-style screening report.")

    # ----- Questionnaire form -----
    st.subheader("Health Questionnaire")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["male", "female", "other"], index=0)
        smoking = st.checkbox("Smoking history")
        wheezing = st.checkbox("Wheezing")
        mucus = st.checkbox("Mucus / phlegm")
    with col2:
        night_cough = st.checkbox("Night-time cough")
        frequent_cough = st.checkbox("Frequent cough")
        difficulty_breathing = st.checkbox("Difficulty breathing")
        family_asthma = st.checkbox("Family history of asthma")
        smoke_exposure = st.checkbox("Secondhand smoke exposure")

    questionnaire = {
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "wheezing": wheezing,
        "mucus": mucus,
        "night_cough": night_cough,
        "frequent_cough": frequent_cough,
        "difficulty_breathing": difficulty_breathing,
        "family_asthma": family_asthma,
        "smoke_exposure": smoke_exposure,
    }

    # ----- Audio Recording -----
    st.subheader("Audio Recording")
    wav_file = st.file_uploader(
        "Upload WAV file (multiple coughs in one session)",
        type=["wav"],
        key="upload_wav",
    )

    wav_bytes = None
    if wav_file is not None:
        wav_bytes = wav_file.read()
        st.audio(wav_bytes, format="audio/wav")

    st.markdown("---")

    st.markdown("---")
    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if wav_bytes is None:
            st.error("Please upload a WAV file before analyzing.")
            return
        with st.spinner("Analyzing cough audio and generating report..."):
            result = run_pipeline(wav_bytes, questionnaire)

        if result.get("error"):
            st.error(result["error"])
            if result.get("cough_count", 0) == 0:
                st.info("Tip: Use a recording with clear cough sounds and minimal background noise.")
            return

        # Summary metrics
        st.subheader("Screening Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total coughs", result["cough_count"])
        with c2:
            st.metric("Wet coughs", result["wet_count"])
        with c3:
            st.metric("Lung health index", f"{result['lung_health_index']:.1f}")
        with c4:
            st.metric("Overall severity", result["severity"])

        st.metric("Wetness %", f"{result['wet_percentage']:.1f}%")
        st.markdown("**Asthma:** " + result["asthma_risk"] + f" (p={result['asthma_probability']:.2f})  |  **COPD:** " + result["copd_risk"] + f" (p={result['copd_probability']:.2f})")

        # Full report
        st.subheader("Clinical Report")
        st.markdown(f'<div class="report-box">{result["report_text"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

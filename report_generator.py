"""
report_generator.py
-------------------
Clinical-style respiratory screening report from pipeline outputs.
Includes lung health index computation and severity classification.
"""

import math


def compute_lung_health_index(
    wet_percentage: float,
    cough_count: int,
    copd_probability: float,
    asthma_probability: float,
    wet_penalty_weight: float = 0.15,
    cough_penalty_weight: float = 0.5,
    copd_penalty_weight: float = 25.0,
    asthma_penalty_weight: float = 15.0,
) -> float:
    """
    Lung health index 0-100. Start at 100, subtract weighted penalties.
    Clamp to [0, 100].
    """
    score = 100.0
    # Wet percentage: 0-100 -> penalty up to wet_penalty_weight * 100
    score -= (wet_percentage / 100.0) * (wet_penalty_weight * 100)
    # Cough count: cap penalty (e.g. 20 coughs = max penalty)
    cough_penalty = min(cough_count * cough_penalty_weight, 20 * cough_penalty_weight)
    score -= cough_penalty
    # COPD prob 0-1 -> penalty up to copd_penalty_weight
    score -= copd_probability * copd_penalty_weight
    # Asthma prob
    score -= asthma_probability * asthma_penalty_weight
    return max(0.0, min(100.0, round(score, 1)))


def overall_severity(
    lung_health_index: float,
    asthma_prob: float,
    copd_prob: float,
) -> str:
    """
    Overall abnormality severity from index and probabilities.
    """
    if lung_health_index >= 75 and asthma_prob < 0.4 and copd_prob < 0.4:
        return "Normal"
    if lung_health_index >= 50:
        return "Mild"
    if lung_health_index >= 25:
        return "Moderate"
    return "Severe"


def generate_report(
    patient_metadata: dict,
    cough_stats: dict,
    wet_pct: float,
    lung_health_index: float,
    asthma_prob: float,
    asthma_risk: str,
    copd_prob: float,
    copd_risk: str,
    healthy_prob: float,
    severity: str,
) -> str:
    """
    Generate structured medical-style text report.
    """
    lines = [
        "---------------------------------------",
        "   RESPIRATORY SCREENING REPORT",
        "---------------------------------------",
        "",
        "PATIENT METADATA",
        "---------------------------------------",
        f"  Age:              {patient_metadata.get('age', 'N/A')}",
        f"  Gender:           {patient_metadata.get('gender', 'N/A')}",
        "",
        "COUGH STATISTICS",
        "---------------------------------------",
        f"  Total coughs:     {cough_stats.get('cough_count', 0)}",
        f"  Wet coughs:       {cough_stats.get('wet_count', 0)}",
        f"  Dry coughs:       {cough_stats.get('dry_count', 0)}",
        f"  Wetness %:        {wet_pct:.1f}%",
        "",
        "LUNG HEALTH INDEX",
        "---------------------------------------",
        f"  Score (0-100):    {lung_health_index:.1f}",
        "",
        "ASTHMA RISK",
        "---------------------------------------",
        f"  Probability:      {asthma_prob:.2f}",
        f"  Risk level:       {asthma_risk}",
        "",
        "COPD RISK",
        "---------------------------------------",
        f"  Probability:      {copd_prob:.2f}",
        f"  Risk level:       {copd_risk}",
        "",
        "HEALTHY (NO CONDITION) PROBABILITY",
        "---------------------------------------",
        f"  Probability:      {healthy_prob:.2f}",
        "",
        "OVERALL SEVERITY",
        "---------------------------------------",
        f"  Classification:   {severity}",
        "",
        "---------------------------------------",
        "   END OF REPORT",
        "---------------------------------------",
    ]
    return "\n".join(lines)


def build_report_from_pipeline(
    questionnaire: dict,
    cough_count: int,
    wet_count: int,
    dry_count: int,
    wet_percentage: float,
    disease_proba: dict,
    risk_levels: dict,
) -> str:
    """
    Build full report from pipeline outputs.
    disease_proba: { asthma, copd, healthy } -> probability
    risk_levels: { asthma, copd } -> "Low"|"Medium"|"High"
    """
    patient_metadata = {
        "age": questionnaire.get("age", "N/A"),
        "gender": questionnaire.get("gender", "N/A"),
    }
    cough_stats = {
        "cough_count": cough_count,
        "wet_count": wet_count,
        "dry_count": dry_count,
    }
    asthma_prob = disease_proba.get("asthma", 0.0)
    copd_prob = disease_proba.get("copd", 0.0)
    healthy_prob = disease_proba.get("healthy", 0.0)
    asthma_risk = risk_levels.get("asthma", "Low")
    copd_risk = risk_levels.get("copd", "Low")

    lung_health_index = compute_lung_health_index(
        wet_percentage, cough_count, copd_prob, asthma_prob
    )
    severity = overall_severity(lung_health_index, asthma_prob, copd_prob)

    return generate_report(
        patient_metadata=patient_metadata,
        cough_stats=cough_stats,
        wet_pct=wet_percentage,
        lung_health_index=lung_health_index,
        asthma_prob=asthma_prob,
        asthma_risk=asthma_risk,
        copd_prob=copd_prob,
        copd_risk=copd_risk,
        healthy_prob=healthy_prob,
        severity=severity,
    )

# Datasets for Cough-Based Respiratory Screening

Public datasets you can use to train or evaluate the cough detection, wet/dry classification, and respiratory disease (e.g. asthma/COPD) models in this project.

---

## 1. **COUGHVID** (best fit for this project)

- **What:** Large crowdsourced cough audio dataset with **expert labels** including **wet/dry** and respiratory abnormalities.
- **Size:** 20,000–30,000+ cough recordings (V3).
- **Labels:** 2,000+ recordings labeled by pulmonologists: cough type (wet/dry), dyspnea, wheezing, stridor, respiratory tract infection, obstructive disease, COVID-19.
- **Format:** Audio (WAV) + metadata CSV.
- **License:** CC BY 4.0.

**Download:**

- **Zenodo (v2):** https://zenodo.org/records/4498364  
- **Zenodo (v3):** https://zenodo.org/records/7024894  
- **Kaggle (V3):** https://www.kaggle.com/datasets/orvile/coughvid-v3  

**Use in this project:** Train wet/dry classifier from “wet/dry” labels; use “respiratory condition” / “obstructive disease” style labels as proxies for asthma/COPD if you map them to your questionnaire + audio pipeline.

---

## 2. **COVID-19 Sounds**

- **What:** Multi-modal respiratory audio (breathing, cough, voice) for symptom and COVID-19 prediction.
- **Size:** 53,449 samples (~552 hours), 36,116 participants; 2,106 COVID-19 positive.
- **Labels:** Self-reported COVID-19 status and respiratory symptoms.
- **Use:** Respiratory symptom prediction (ROC-AUC > 0.7); can be adapted for cough-based screening.

**Reference:** Cambridge University; check for current download/access via official COVID-19 Sounds project pages.

---

## 3. **Coswara**

- **What:** Respiratory sounds (breathing, cough, vowels, speech) + demographics and health/symptom metadata.
- **Size:** 2,635 individuals, ~65 hours; 674 COVID-19 positive, 1,819 negative, 142 recovered.
- **Labels:** Demographics (age, gender), symptoms, pre-existing respiratory ailments, COVID-19 status; manually checked quality.
- **Use:** Cough + questionnaire-style metadata; good for fusion models (audio + demographics/symptoms). Not asthma/COPD-specific but useful for “respiratory abnormality” screening.

**Reference:** Nature Scientific Data, 2023 — “Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection.”

---

## 4. **NoCoCoDa**

- **What:** Cough events from COVID-19 patient interviews; **productive (wet) vs dry** cough labels.
- **Size:** 73 cough events.
- **Labels:** Cough phase and wet vs dry (productive) classification.
- **Use:** Small but focused on wet/dry; request access from the research team.

---

## Summary for this project

| Need              | Recommended dataset | Why                                      |
|-------------------|---------------------|------------------------------------------|
| Wet/dry cough     | **COUGHVID**        | Expert wet/dry labels; large scale        |
| Respiratory label | **COUGHVID**        | Pulmonologist labels, obstructive etc.   |
| Audio + metadata  | **Coswara**         | Demographics + symptoms, fusion-friendly |
| Small wet/dry     | **NoCoCoDa**        | Explicit productive vs dry               |

**Practical path:** Start with **COUGHVID** (Kaggle or Zenodo): use the metadata CSV to filter by “wet”/“dry” and by respiratory/obstructive flags, then align with this project’s `train_models.py` and `classifier.py` (wet/dry + disease) as in the README.

import json
import pandas as pd

from src.nlp.extractor import extract_entities


# ==========================================
# SELECT CONSULTATION
# ==========================================

consultation_id = "C00002"


# ==========================================
# LOAD GOLD-STANDARD REFERENCE
# ==========================================

with open(
    "evaluation/gold_standard.json",
    "r",
    encoding="utf-8"
) as file:
    gold_data = json.load(file)

reference = gold_data[consultation_id]


# ==========================================
# LOAD TRANSCRIPT
# ==========================================

df = pd.read_excel(
    "data/processed/01_Consultation_Transcription.xlsx"
)

consultation = df[
    df["consultation_id"].astype(str).str.strip()
    == consultation_id
].iloc[0]

transcript = consultation["transcript"]


# ==========================================
# GENERATE MODEL OUTPUT
# ==========================================

model_output = extract_entities(
    transcript,
    engine="auto"
)


# ==========================================
# NORMALIZE REFERENCE SYMPTOMS
# CORE SKIN SYMPTOMS ONLY
# ==========================================

reference_symptoms = set()

for symptom in reference.get("symptoms", []):

    text = symptom.get("text", "").lower()

    if "itch" in text:
        reference_symptoms.add("itching")

    if "sore" in text or "pain" in text:
        reference_symptoms.add("skin soreness")

    if "red" in text:
        reference_symptoms.add("redness")

    if "crack" in text:
        reference_symptoms.add("cracked skin")


# ==========================================
# NORMALIZE MODEL SYMPTOMS
# CORE SKIN SYMPTOMS ONLY
# ==========================================

model_symptoms = set()

for symptom in model_output.get("symptoms", []):

    text = symptom.get("text", "").lower()

    if "itch" in text:
        model_symptoms.add("itching")

    if "sore" in text or "pain" in text:
        model_symptoms.add("skin soreness")

    if "red" in text:
        model_symptoms.add("redness")

    if "crack" in text:
        model_symptoms.add("cracked skin")


# ==========================================
# CALCULATE METRICS
# ==========================================

true_positives = len(
    reference_symptoms & model_symptoms
)

false_positives = len(
    model_symptoms - reference_symptoms
)

false_negatives = len(
    reference_symptoms - model_symptoms
)


precision = (
    true_positives /
    (true_positives + false_positives)
    if true_positives + false_positives > 0
    else 0
)


recall = (
    true_positives /
    (true_positives + false_negatives)
    if true_positives + false_negatives > 0
    else 0
)


f1_score = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===== NORMALIZED EVALUATION =====")

print("Consultation:", consultation_id)

print("\nReference symptoms:", reference_symptoms)

print("Model symptoms:", model_symptoms)

print("\nTrue Positives:", true_positives)

print("False Positives:", false_positives)

print("False Negatives:", false_negatives)

print("\nPrecision:", round(precision, 2))

print("Recall:", round(recall, 2))

print("F1-score:", round(f1_score, 2))
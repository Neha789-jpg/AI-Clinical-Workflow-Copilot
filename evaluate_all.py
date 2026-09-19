import json
import pandas as pd

from src.nlp.extractor import extract_entities


# ==========================================
# FILE PATHS
# ==========================================

REFERENCE_FILE = "evaluation/auto_gold_standard.json"

TRANSCRIPT_FILE = (
    "data/processed/01_Consultation_Transcription.xlsx"
)


# ==========================================
# LOAD REFERENCE DATA
# ==========================================

with open(
    REFERENCE_FILE,
    "r",
    encoding="utf-8"
) as file:

    gold_data = json.load(file)


# ==========================================
# LOAD TRANSCRIPTS
# ==========================================

df = pd.read_excel(TRANSCRIPT_FILE)


# ==========================================
# NORMALIZATION FUNCTION
# ==========================================

def normalize_symptoms(symptoms):

    normalized = set()

    for symptom in symptoms:

        if isinstance(symptom, dict):

            text = symptom.get("text", "").lower()

        else:

            text = str(symptom).lower()

        if "itch" in text:
            normalized.add("itching")

        if "sore" in text or "pain" in text:
            normalized.add("skin soreness")

        if "red" in text:
            normalized.add("redness")

        if "crack" in text:
            normalized.add("cracked skin")

        if "fever" in text:
            normalized.add("fever")

        if "vomit" in text:
            normalized.add("vomiting")

        if "diarr" in text:
            normalized.add("diarrhea")

        if "weak" in text:
            normalized.add("weakness")

        if "cough" in text:
            normalized.add("cough")

        if "headache" in text:
            normalized.add("headache")

        if "nause" in text:
            normalized.add("nausea")

    return normalized


# ==========================================
# OVERALL COUNTERS
# ==========================================

total_tp = 0
total_fp = 0
total_fn = 0


# ==========================================
# EVALUATE CONSULTATIONS
# ==========================================

for consultation_id, reference in gold_data.items():

    matching_rows = df[
        df["consultation_id"].astype(str).str.strip()
        == consultation_id
    ]

    if matching_rows.empty:

        print(
            f"Skipping {consultation_id}: "
            "transcript not found"
        )

        continue

    transcript = matching_rows.iloc[0]["transcript"]

    print(
        f"\nEvaluating {consultation_id}..."
    )

    try:

        model_output = extract_entities(
            transcript,
            engine="auto"
        )

        reference_symptoms = normalize_symptoms(
            reference.get("symptoms", [])
        )

        model_symptoms = normalize_symptoms(
            model_output.get("symptoms", [])
        )

        tp = len(
            reference_symptoms & model_symptoms
        )

        fp = len(
            model_symptoms - reference_symptoms
        )

        fn = len(
            reference_symptoms - model_symptoms
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            "Reference:",
            reference_symptoms
        )

        print(
            "Model:",
            model_symptoms
        )

        print(
            f"TP: {tp} | FP: {fp} | FN: {fn}"
        )

    except Exception as error:

        print(
            f"Error processing {consultation_id}: "
            f"{error}"
        )


# ==========================================
# OVERALL METRICS
# ==========================================

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

f1_score = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)


# ==========================================
# DISPLAY FINAL RESULTS
# ==========================================

print("\n==========================================")
print("OVERALL EVALUATION RESULTS")
print("==========================================")

print("Consultations evaluated:", len(gold_data))

print("Total True Positives:", total_tp)
print("Total False Positives:", total_fp)
print("Total False Negatives:", total_fn)

print("\nPrecision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1_score, 2))

print("==========================================")
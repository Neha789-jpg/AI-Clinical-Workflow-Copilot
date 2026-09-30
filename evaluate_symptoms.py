import json
import re
import os
import pandas as pd

from openai import OpenAI
from src.nlp.extractor import extract_with_llm


# ============================================================
# 1. FILES / CONFIG
# ============================================================

INPUT_FILE = "data/processed/01_Consultation_Transcription.xlsx"
OUTPUT_FILE = "evaluation/results/symptom_llm_results.csv"

# None = all consultations
with open("evaluation/auto_gold_standard.json", "r", encoding="utf-8") as f:
    gold_ids = set(json.load(f).keys())
TARGET_IDS = [f"C{i:05d}" for i in range(1, 58) if f"C{i:05d}" in gold_ids]
GROQ_API_KEY = None


# ============================================================
# 2. ORIGINAL REFERENCE PARSER
# ============================================================

def parse_reference_symptoms(value):
    """
    Convert the existing symptoms column into a list.

    The original reference is preserved for comparison,
    but is NOT assumed to be a complete gold standard.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            results = []

            for item in parsed:
                if isinstance(item, dict):
                    symptom_text = item.get("text")

                    if symptom_text:
                        results.append(str(symptom_text))
                else:
                    results.append(str(item))

            return results

    except Exception:
        pass

    return [text]


# ============================================================
# 3. GET PREDICTED SYMPTOMS FROM OUR EXTRACTOR
# ============================================================

def parse_predicted_symptoms(result):
    """
    Extract symptom text from the real LLM extractor output.
    """

    symptoms = result.get("symptoms", [])

    if not isinstance(symptoms, list):
        return []

    extracted = []

    for symptom in symptoms:

        if isinstance(symptom, dict):
            text = symptom.get("text")

            if text:
                extracted.append(str(text))

        elif isinstance(symptom, str):
            extracted.append(symptom)

    return extracted


# ============================================================
# 4. SIMPLE NORMALIZATION
# ============================================================

def simple_normalize(text):
    """
    Light normalization for the original reference metrics.

    Semantic matching is handled by the evaluator LLM.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# 5. TRANSCRIPT-GROUNDED EVALUATOR PROMPT
# ============================================================

EVALUATOR_PROMPT = """
You evaluate a clinical symptom extraction system.

Judge the extractor ONLY against the consultation transcript.

The original dataset reference may be incomplete. Therefore:
- Do NOT penalize a prediction merely because it is absent from the reference.
- A symptom explicitly supported by the transcript is valid.
- A prediction unsupported by the transcript is UNSUPPORTED.
- Do not infer symptoms from diagnoses, medications, tests, or context.
- Do not treat doctor questions, denied symptoms, or hypothetical symptoms as present.

Identify ALL distinct clinically relevant symptoms explicitly supported
by the transcript.

Different wording with the same clinical meaning should match.

Do NOT merge genuinely different symptoms.
For example:
headache != nausea
headache != visual disturbance
abdominal pain != nausea

A symptom that occurred previously can still be transcript-supported,
even if it is no longer current.

For every predicted symptom, return exactly one:
MATCHED or UNSUPPORTED.

A prediction is MATCHED if the transcript clearly supports it.

A symptom is MISSED only when:
1. it is clearly supported by the transcript, and
2. no semantically equivalent predicted symptom exists.

Also evaluate each original reference symptom as:
SUPPORTED, NOT_SUPPORTED, or PARAPHRASE_OF_SUPPORTED.

Use short canonical clinical labels.

Return ONLY valid JSON in exactly this structure:

{
  "transcript_supported_symptoms": [
    {
      "text": "...",
      "canonical": "..."
    }
  ],
  "predicted_evaluations": [
    {
      "predicted": "...",
      "canonical": "...",
      "status": "MATCHED|UNSUPPORTED",
      "matched_transcript_symptom": "...",
      "reason": "..."
    }
  ],
  "missed_symptoms": [
    {
      "text": "...",
      "canonical": "...",
      "reason": "..."
    }
  ],
  "reference_evaluations": [
    {
      "reference": "...",
      "canonical": "...",
      "status": "SUPPORTED|NOT_SUPPORTED|PARAPHRASE_OF_SUPPORTED",
      "matched_transcript_symptom": "...",
      "reason": "..."
    }
  ]
}
"""


# ============================================================
# 6. GROQ CLIENT
# ============================================================

def get_llm_client():

    global GROQ_API_KEY

    if GROQ_API_KEY is None:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )


# ============================================================
# 7. LLM EVALUATOR
# ============================================================

def evaluate_with_llm(
    transcript,
    reference_symptoms,
    predicted_symptoms
):
    """
    Use an LLM as a transcript-grounded evaluator.

    Important:
    Rate-limit errors are NOT retried because retrying them
    consumes/wastes more requests while the limit is active.
    """

    client = get_llm_client()

    payload = {
        "transcript": transcript,
        "original_reference_symptoms": reference_symptoms,
        "predicted_symptoms": predicted_symptoms
    }

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": EVALUATOR_PROMPT
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False
                    )
                }
            ]
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:

        error_text = str(e)

        # Do not retry Groq rate-limit errors
        if (
            "429" in error_text
            or "rate_limit_exceeded" in error_text
            or "tokens per day" in error_text
        ):
            raise RuntimeError(
                "GROQ_RATE_LIMIT: " + error_text
            )

        raise


# ============================================================
# 8. SAFE LIST
# ============================================================

def safe_list(data, key):

    value = data.get(key, [])

    if isinstance(value, list):
        return value

    return []


# ============================================================
# 9. CALCULATE TRANSCRIPT METRICS
# ============================================================

def calculate_transcript_metrics(evaluation):

    predicted_evaluations = safe_list(
        evaluation,
        "predicted_evaluations"
    )

    missed_symptoms = safe_list(
        evaluation,
        "missed_symptoms"
    )

    tp = sum(
        1
        for item in predicted_evaluations
        if item.get("status") == "MATCHED"
    )

    fp = sum(
        1
        for item in predicted_evaluations
        if item.get("status") == "UNSUPPORTED"
    )

    fn = len(missed_symptoms)

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return tp, fp, fn, precision, recall, f1


# ============================================================
# 10. LOAD INPUT
# ============================================================

df = pd.read_excel(INPUT_FILE)

if TARGET_IDS is None:

    eval_df = df.copy()

else:

    eval_df = df[
        df["consultation_id"].isin(TARGET_IDS)
    ].copy()


# ============================================================
# 11. RESUME PREVIOUS RESULTS
# ============================================================

existing_results = []

if os.path.exists(OUTPUT_FILE):

    try:

        previous_df = pd.read_csv(
            OUTPUT_FILE
        )

        # IMPORTANT:
        # Keep only results belonging to the consultations
        # selected from auto_gold_standard.json.
        previous_df = previous_df[
            previous_df["consultation_id"].astype(str).isin(TARGET_IDS)
        ].copy()

        existing_results = previous_df.to_dict(
            "records"
        )

        completed_ids = {
            str(row["consultation_id"])
            for row in existing_results
            if row.get("status") == "EVALUATED"
        }

        print(
            f"\nPrevious results belonging to auto-gold set: "
            f"{len(existing_results)} rows"
        )

        print(
            f"Already evaluated successfully: "
            f"{len(completed_ids)} consultations"
        )

        print(
            "These consultations will be skipped.\n"
        )

    except Exception as e:

        print(
            f"\nCould not read previous results: {e}"
        )

        existing_results = []
        completed_ids = set()

else:

    completed_ids = set()


# ============================================================
# 12. RESULTS STORAGE
# ============================================================

results = existing_results.copy()

# Original reference metrics
old_total_tp = 0
old_total_fp = 0
old_total_fn = 0

# Transcript-grounded metrics
total_tp = 0
total_fp = 0
total_fn = 0


# Recalculate metrics from already completed results
for r in results:

    if r.get("status") != "EVALUATED":
        continue

    try:

        total_tp += int(r.get("tp", 0))
        total_fp += int(r.get("fp", 0))
        total_fn += int(r.get("fn", 0))

        old_total_tp += int(
            r.get("original_reference_tp", 0)
        )

        old_total_fp += int(
            r.get("original_reference_fp", 0)
        )

        old_total_fn += int(
            r.get("original_reference_fn", 0)
        )

    except Exception:
        pass


# ============================================================
# 13. START
# ============================================================

print("\n==============================================")
print("TRANSCRIPT-GROUNDED LLM SYMPTOM EVALUATION")
print("==============================================\n")

print(
    f"Consultations selected: {len(eval_df)}"
)

print(
    f"Already completed: {len(completed_ids)}"
)

print(
    f"Remaining: "
    f"{len(eval_df) - len(completed_ids)}"
)

print()


# ============================================================
# 14. MAIN LOOP
# ============================================================

for _, row in eval_df.iterrows():

    consultation_id = str(
        row["consultation_id"]
    )

    # --------------------------------------------------------
    # Skip already completed consultations
    # --------------------------------------------------------

    if consultation_id in completed_ids:

        print(
            f"Skipping {consultation_id} "
            f"(already evaluated)"
        )

        continue


    transcript = str(
        row["transcript"]
    )

    reference_raw = parse_reference_symptoms(
        row["symptoms"]
    )

    print(
        f"Evaluating {consultation_id}..."
    )


    try:

        # ====================================================
        # STEP 1: REAL EXTRACTOR
        # ====================================================

        llm_output = extract_with_llm(
            transcript,
            retries=1
        )

        predicted_raw = parse_predicted_symptoms(
            llm_output
        )


        # ====================================================
        # STEP 2: ORIGINAL REFERENCE METRICS
        # ====================================================

        reference_normalized = {
            simple_normalize(x)
            for x in reference_raw
            if simple_normalize(x)
        }

        predicted_normalized = {
            simple_normalize(x)
            for x in predicted_raw
            if simple_normalize(x)
        }

        old_tp = len(
            reference_normalized &
            predicted_normalized
        )

        old_fp = len(
            predicted_normalized -
            reference_normalized
        )

        old_fn = len(
            reference_normalized -
            predicted_normalized
        )

        old_total_tp += old_tp
        old_total_fp += old_fp
        old_total_fn += old_fn


        # ====================================================
        # STEP 3: TRANSCRIPT-GROUNDED EVALUATOR
        # ====================================================

        evaluation = evaluate_with_llm(
            transcript=transcript,
            reference_symptoms=reference_raw,
            predicted_symptoms=predicted_raw
        )


        (
            tp,
            fp,
            fn,
            precision,
            recall,
            f1
        ) = calculate_transcript_metrics(
            evaluation
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn


        supported = safe_list(
            evaluation,
            "transcript_supported_symptoms"
        )

        predicted_evaluations = safe_list(
            evaluation,
            "predicted_evaluations"
        )

        missed = safe_list(
            evaluation,
            "missed_symptoms"
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            "\n  Original reference:"
        )

        print(
            f"    {reference_raw}"
        )

        print(
            "\n  Extractor predicted:"
        )

        print(
            f"    {predicted_raw}"
        )

        print(
            "\n  Transcript-supported:"
        )

        print(
            "   ",
            [
                item.get("text")
                for item in supported
            ]
        )

        print(
            "\n  Predicted evaluation:"
        )

        for item in predicted_evaluations:

            print(
                f"    [{item.get('status')}] "
                f"{item.get('predicted')}"
            )

            if item.get(
                "matched_transcript_symptom"
            ):

                print(
                    "       ->",
                    item.get(
                        "matched_transcript_symptom"
                    )
                )

        print(
            "\n  Missed:"
        )

        print(
            "   ",
            [
                item.get("text")
                for item in missed
            ]
        )

        print(
            f"\n  Transcript-grounded:"
            f" P={precision:.2f}"
            f" R={recall:.2f}"
            f" F1={f1:.2f}"
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append({

            "consultation_id":
                consultation_id,

            "status":
                "EVALUATED",

            "original_reference":
                " | ".join(reference_raw),

            "extractor_predictions":
                " | ".join(predicted_raw),

            "original_reference_tp":
                old_tp,

            "original_reference_fp":
                old_fp,

            "original_reference_fn":
                old_fn,

            "transcript_supported_symptoms":
                " | ".join(
                    item.get("text", "")
                    for item in supported
                ),

            "transcript_supported_canonical":
                " | ".join(
                    item.get("canonical", "")
                    for item in supported
                ),

            "matched_predictions":
                " | ".join(
                    item.get("predicted", "")
                    for item in predicted_evaluations
                    if item.get("status") == "MATCHED"
                ),

            "unsupported_predictions":
                " | ".join(
                    item.get("predicted", "")
                    for item in predicted_evaluations
                    if item.get("status") == "UNSUPPORTED"
                ),

            "missed_symptoms":
                " | ".join(
                    item.get("text", "")
                    for item in missed
                ),

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "evaluator_json":
                json.dumps(
                    evaluation,
                    ensure_ascii=False
                )
        })


        # ----------------------------------------------------
        # SAVE AFTER EVERY SUCCESSFUL CONSULTATION
        # ----------------------------------------------------

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )


    except Exception as e:

        print(
            f"\n  NOT EVALUATED: {e}\n"
        )

        # If Groq rate limit is reached, STOP immediately.
        # There is no point attempting the remaining consultations.
        if "GROQ_RATE_LIMIT" in str(e):

            print(
                "Groq rate limit reached."
            )

            print(
                "Stopping safely. "
                "Successful results have already been saved."
            )

            break


# ============================================================
# 15. FINAL RESULTS
# ============================================================

evaluated = [
    r
    for r in results
    if r.get("status") == "EVALUATED"
]

not_evaluated = [
    r
    for r in results
    if r.get("status") == "NOT_EVALUATED"
]


print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    f"Total consultations selected : "
    f"{len(eval_df)}"
)

print(
    f"LLM evaluated                : "
    f"{len(evaluated)}"
)

print(
    f"Not evaluated                : "
    f"{len(not_evaluated)}"
)


# ============================================================
# 16. TRANSCRIPT-GROUNDED METRICS
# ============================================================

if evaluated:

    micro_precision = (
        total_tp /
        (total_tp + total_fp)
        if (total_tp + total_fp)
        else 0
    )

    micro_recall = (
        total_tp /
        (total_tp + total_fn)
        if (total_tp + total_fn)
        else 0
    )

    micro_f1 = (
        2 *
        micro_precision *
        micro_recall /
        (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0
    )

    print(
        "\n----------------------------------------------"
    )

    print(
        "TRANSCRIPT-GROUNDED METRICS"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"True Positives  : {total_tp}"
    )

    print(
        f"False Positives : {total_fp}"
    )

    print(
        f"False Negatives : {total_fn}"
    )

    print(
        f"\nPrecision       : {micro_precision:.3f}"
    )

    print(
        f"Recall          : {micro_recall:.3f}"
    )

    print(
        f"F1              : {micro_f1:.3f}"
    )


    # ========================================================
    # ORIGINAL REFERENCE METRICS
    # ========================================================

    old_precision = (
        old_total_tp /
        (old_total_tp + old_total_fp)
        if (old_total_tp + old_total_fp)
        else 0
    )

    old_recall = (
        old_total_tp /
        (old_total_tp + old_total_fn)
        if (old_total_tp + old_total_fn)
        else 0
    )

    old_f1 = (
        2 *
        old_precision *
        old_recall /
        (old_precision + old_recall)
        if (old_precision + old_recall)
        else 0
    )

    print(
        "\n----------------------------------------------"
    )

    print(
        "ORIGINAL REFERENCE METRICS"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Precision       : {old_precision:.3f}"
    )

    print(
        f"Recall          : {old_recall:.3f}"
    )

    print(
        f"F1              : {old_f1:.3f}"
    )

else:

    print(
        "\nNo consultations were successfully evaluated."
    )


# ============================================================
# 17. SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)
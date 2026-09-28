
# Step 5 - suggest SNOMED CT codes for the diagnoses in a consultation.
#
# Important rule from the dataset plan: never invent SNOMED identifiers.
# So the codes ONLY ever come from data/reference/snomed_common.csv.
# The LLM (if available) just picks which row matches the diagnosis text;
# without an LLM we fall back to fuzzy string matching.

import os
import json
import difflib
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REFERENCE_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "snomed_common.csv"


def load_reference():
    df = pd.read_csv(REFERENCE_CSV)
    df["synonyms"] = df["synonyms"].fillna("")
    return df


REFERENCE = load_reference()


# ----------------------------------------------------------
# Rules version: fuzzy text matching
# ----------------------------------------------------------

def _all_terms():
    """Build a list of (term, row_index) for every diagnosis and synonym."""
    terms = []
    for i, row in REFERENCE.iterrows():
        terms.append((row["diagnosis"].lower(), i))
        for syn in str(row["synonyms"]).split(";"):
            syn = syn.strip().lower()
            if syn:
                terms.append((syn, i))
    return terms


TERMS = _all_terms()


def match_rules(diagnosis_text):
    """Return the best matching reference row for a diagnosis, or None."""
    text = diagnosis_text.lower().strip()

    # 1. exact or contained match first
    for term, i in TERMS:
        if term == text or term in text:
            return i, 1.0

    # 2. fuzzy match
    best_i, best_score = None, 0.0
    for term, i in TERMS:
        score = difflib.SequenceMatcher(None, text, term).ratio()
        if score > best_score:
            best_i, best_score = i, score

    if best_score >= 0.75:
        return best_i, best_score
    return None, best_score


# ----------------------------------------------------------
# LLM version: model picks a row, never writes a code itself
# ----------------------------------------------------------

PROMPT = """You are a clinical coder for the NHS.
You will get a diagnosis phrase from a GP consultation and a numbered list of
candidate SNOMED CT concepts. Pick the single best matching candidate.

Return ONLY JSON: {"choice": <candidate number>, "confidence": "high|medium|low"}
If none of the candidates is a reasonable match, return {"choice": null, "confidence": "low"}.
Never suggest a code that is not in the list."""


def match_llm(diagnosis_text):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1")

    candidates = "\n".join(
        f"{i}. {row['snomed_description']} ({row['diagnosis']})"
        for i, row in REFERENCE.iterrows()
    )
    user_message = f"DIAGNOSIS: {diagnosis_text}\n\nCANDIDATES:\n{candidates}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": user_message}],
    )
    data = json.loads(response.choices[0].message.content)
    choice = data.get("choice")
    if choice is None or int(choice) not in REFERENCE.index:
        return None, "low"
    return int(choice), data.get("confidence", "medium")


# ----------------------------------------------------------
# Main function
# ----------------------------------------------------------

def suggest_code(diagnosis_text, engine="auto"):
    """Suggest one SNOMED code for a single diagnosis phrase."""
    use_llm = engine == "llm" or (engine == "auto" and os.getenv("GROQ_API_KEY"))

    row_index, confidence = None, "low"
    if use_llm:
        try:
            row_index, confidence = match_llm(diagnosis_text)
        except Exception as e:
            print("[coding] LLM failed, using rules:", e)
            use_llm = False
    if not use_llm or row_index is None:
        row_index, score = match_rules(diagnosis_text)
        confidence = "high" if score >= 0.95 else "medium" if score >= 0.75 else "low"

    if row_index is None:
        return {"diagnosis": diagnosis_text, "snomed_code": None,
                "snomed_description": None, "confidence": "low",
                "note": "No match in reference table - code manually"}

    row = REFERENCE.loc[row_index]
    return {"diagnosis": diagnosis_text,
            "snomed_code": str(row["snomed_code"]),
            "snomed_description": row["snomed_description"],
            "confidence": confidence}


def suggest_codes(entities, engine="auto"):
    """Suggest codes for every diagnosis in the extracted entities."""
    results = []
    for d in entities.get("diagnoses", []):
        text = d.get("text") if isinstance(d, dict) else str(d)
        if text:
            result = suggest_code(text, engine=engine)
            result["status"] = d.get("status") if isinstance(d, dict) else None
            results.append(result)
    return results


if __name__ == "__main__":
    # the three examples from the client document
    for dx in ["Diabetes Mellitus", "knee pain", "migraine", "possible viral URTI", "sprained ankle"]:
        print(json.dumps(suggest_code(dx, engine="rules"), indent=2))
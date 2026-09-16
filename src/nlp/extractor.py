import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Vocabulary lists (rule-based engine)
# ---------------------------------------------------------------------------
SYMPTOM_VOCAB = ["headache", "dizziness", "vertigo", "nausea", "vomiting", "diarrhea",
                 "fever", "cough", "chest pain", "back pain", "knee pain", "sore throat",
                 "shortness of breath", "palpitations", "fatigue", "rash", "swelling",
                 "abdominal pain", "stomach pain", "heartburn", "blurred vision"]

DIAGNOSIS_VOCAB = ["gastroenteritis", "asthma", "hypertension", "diabetes", "migraine",
                   "vestibular disorder", "urinary tract infection", "pneumonia",
                   "bronchitis", "tonsillitis", "eczema", "osteoarthritis", "sinusitis",
                   "reflux", "viral infection", "depression", "anxiety"]

MEDICATION_VOCAB = ["paracetamol", "ibuprofen", "amoxicillin", "metformin", "omeprazole",
                    "salbutamol", "cetirizine", "atorvastatin", "ramipril", "amlodipine",
                    "sertraline", "prednisolone", "insulin", "levothyroxine", "dioralyte"]

ALLERGY_VOCAB = ["penicillin", "amoxicillin", "aspirin", "ibuprofen", "latex",
                 "nuts", "peanuts", "shellfish", "eggs"]

HISTORY_VOCAB = ["asthma", "hypertension", "diabetes", "heart failure", "stroke",
                 "cancer", "depression", "epilepsy", "arthritis", "kidney stones"]


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------
def find_terms(text, vocab):
    """Return every word from vocab that appears in text (longest first)."""
    found = []
    low = text.lower()
    for term in sorted(vocab, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            found.append(term)
    return found


def find_duration(text, term):
    """Look for 'for 3 days' / 'two weeks' within 60 chars after a symptom."""
    m = re.search(re.escape(term) + r".{0,60}?(\d+|two|three|four|five|a few|several)\s*"
                  r"(day|week|month|year)s?", text.lower())
    return f"{m.group(1)} {m.group(2)}s" if m else None


def find_vitals(text):
    low = text.lower()
    bp = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", low)
    pulse = re.search(r"(?:pulse|heart rate)\D{0,15}(\d{2,3})", low)
    temp = re.search(r"temp(?:erature)?\D{0,15}(\d{2}(?:\.\d)?)", low)
    spo2 = re.search(r"(?:spo2|sats|oxygen)\D{0,15}(\d{2,3})", low)
    return {
        "bp": f"{bp.group(1)}/{bp.group(2)}" if bp else None,
        "pulse": int(pulse.group(1)) if pulse else None,
        "temperature": float(temp.group(1)) if temp else None,
        "spo2": int(spo2.group(1)) if spo2 else None,
    }


def find_patient_info(text):
    low = text.lower()
    age = re.search(r"\b(\d{1,3})[\s-]*(?:years?[\s-]*old|year old)\b", low)
    gender = None
    if re.search(r"\b(female|woman|she|her)\b", low):
        gender = "female"
    elif re.search(r"\b(male|man|he|his)\b", low):
        gender = "male"
    return {"age": int(age.group(1)) if age else None, "gender": gender}


# ---------------------------------------------------------------------------
# Engine 1: rule-based (offline fallback)
# ---------------------------------------------------------------------------
def extract_with_rules(transcript):
    low = transcript.lower()

    symptoms = [{"text": s, "duration": find_duration(transcript, s)}
                for s in find_terms(transcript, SYMPTOM_VOCAB)]

    history = [h for h in find_terms(transcript, HISTORY_VOCAB)
               if re.search(r"(history|previous|past|diagnosed with|known|have had|i have|i've got)\W{0,60}" + re.escape(h), low)]

    diagnoses = []
    for d in find_terms(transcript, DIAGNOSIS_VOCAB):
        if d in history:
            continue
        suspected = re.search(r"(suspect|likely|possible|probably|think|could be).{0,20}" + re.escape(d), low)
        diagnoses.append({"text": d, "status": "suspected" if suspected else "confirmed"})

    medications = []
    for m in find_terms(transcript, MEDICATION_VOCAB):
        dose = re.search(re.escape(m) + r"\W{0,10}(\d+\s*(?:mg|mcg|g|ml))", low)
        freq = re.search(re.escape(m) + r".{0,40}?(once daily|twice daily|three times a day|as needed|every \d+ hours)", low)
        medications.append({"name": m,
                            "dose": dose.group(1) if dose else None,
                            "frequency": freq.group(1) if freq else None})

    allergies = []
    if "allerg" in low or "nkda" in low:
        allergies = find_terms(transcript, ALLERGY_VOCAB)
        if not allergies and re.search(r"no known|nkda|no allergies", low):
            allergies = ["none known"]

    return {
        "patient_info": find_patient_info(transcript),
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "medications": medications,
        "allergies": allergies,
        "vitals": find_vitals(transcript),
        "medical_history": history,
        "family_history": [],
        "social_history": [],
    }


# ---------------------------------------------------------------------------
# Engine 2: LLM (Groq / OpenAI-compatible)
# ---------------------------------------------------------------------------
PROMPT = """You are a clinical information extraction system.
Read the doctor-patient transcript and return ONLY a JSON object with exactly these keys:
{
  "patient_info": {"age": null, "gender": null},
  "symptoms": [{"text": "...", "duration": null}],
  "diagnoses": [{"text": "...", "status": "confirmed|suspected"}],
  "current_medications": [{"name": "...", "dose": null, "frequency": null}],

  "prescribed": [{"name": "...", "dose": null, "frequency": null}],

  "investigations": ["..."],

  "follow_up": ["..."],

  "advice": ["..."],

  "allergies": ["..."],
  "vitals": {"bp": null, "pulse": null, "temperature": null, "spo2": null},
  "medical_history": ["..."],
  "family_history": ["..."],
  "social_history": ["..."]
}

Rules:

- diagnoses: Include ONLY diagnoses explicitly stated or suggested by the DOCTOR in the transcript.
- Never infer a diagnosis from symptoms alone. If the doctor does not provide a diagnosis, return [].
- Use "suspected" when the doctor hedges (e.g., think, possibly, likely, query).
- Use "confirmed" only when the doctor states the diagnosis as a fact.
- Do not classify a condition as a diagnosis when it is mentioned only as a possible cause, differential consideration, or reason for ordering a test.
- Do not include investigations, tests, test orders, medications, or treatments in the diagnoses list.
- Pregnancy must not be recorded as a diagnosis when it is only mentioned as a possibility, differential consideration, or reason for ordering a pregnancy test.
- Record pregnancy as a diagnosis only when the doctor explicitly identifies it as a suspected or confirmed diagnosis.
- Record "pregnancy test" only under investigations.

- current_medications: Medicines the patient was already taking before this visit.
- prescribed: Medicines the doctor explicitly prescribes, starts, or confirms during this visit.
- A medicine must appear in only one medication list.
- Extract medication names, doses, frequencies, and durations exactly as stated by the doctor.
- Do not change, calculate, round, or assume medication dosages.
- If the dosage is unclear or contradictory, preserve the uncertainty rather than guessing.
- If the doctor corrects, revises, or clarifies a medication instruction, use the final corrected instruction.
- Ignore earlier medication instructions that are subsequently corrected.
- Never combine conflicting dosage statements.
- If the final dosage remains ambiguous, mark it as unclear rather than guessing.
- Medicines mentioned only as general options, examples, or suggestions must not be added to "prescribed"; record them under "advice" instead.

- investigations: Tests or investigations the doctor recommends or orders, such as blood tests, stool tests, pregnancy tests, or scans.
- Keep investigations and tests ONLY in the "investigations" list.
- A test mentioned to rule out a possible condition must be recorded under "investigations", not "diagnoses".

- follow_up: Follow-up instructions explicitly mentioned by the doctor, such as returning after a few days or coming back if symptoms worsen.
- advice: Instructions explicitly given to the patient, such as rest, hydration, diet, or lifestyle advice.
- Keep medicines in "prescribed", tests in "investigations", and patient instructions in "advice" or "follow_up".

- medical_history: Past illnesses, operations, and previous similar episodes.
- family_history: Illnesses in relatives.
- social_history: Smoking, alcohol, occupation, living situation, and sexual health if relevant.
- allergies: If the patient explicitly denies allergies or says "no allergies" / "NKDA", return ["none known"]. Use [] only when allergies are not mentioned.
- gender: Include only if explicitly stated or unambiguous. Menstruation or pregnancy indicates female; otherwise return null.
- Only include information actually stated in the transcript.
- Do not invent, assume, calculate, or infer information.
- Use null for missing single values and [] for missing lists.
- No markdown, no explanation."""

def extract_with_llm(transcript, retries=2):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1")
    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": transcript}],
            )
            data = json.loads(response.choices[0].message.content)

# Safety check: prevent pregnancy test from becoming
# an inferred pregnancy diagnosis.
            prescribed_items = data.get("prescribed", [])

            has_pregnancy_test = any(
               "pregnancy test" in str(item).lower()
               for item in prescribed_items
)

            if has_pregnancy_test:

               data["diagnoses"] = [
                   diagnosis
                   for diagnosis in data.get("diagnoses", [])
                   if diagnosis.get("text", "").lower() != "pregnancy"
    ]

            return data
        except Exception as e:
            last_error = e
            print(f"[extractor] attempt {attempt + 1} failed, retrying...")
    raise last_error


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_entities(transcript, engine="auto"):
    """
    engine = "auto"  -> LLM if GROQ_API_KEY exists, else rules
             "llm"   -> force LLM
             "rules" -> force rule-based
    """
    if not transcript or not transcript.strip():
        return extract_with_rules("")
    if engine == "rules" or (engine == "auto" and not os.getenv("GROQ_API_KEY")):
        return extract_with_rules(transcript)
    try:
        return extract_with_llm(transcript)
    except Exception as e:
        print(f"[extractor] LLM failed ({e}), using rules")
        return extract_with_rules(transcript)


if __name__ == "__main__":
    sample = ("Doctor: How can I help? Patient: I'm a 34 year old woman, I've had really bad "
              "diarrhea for the last 3 days and some stomach pain, and a fever on the first day. "
              "I have asthma and use salbutamol. No known allergies. "
              "Doctor: BP is 120/80, pulse 88, temperature 37.2. I think this is gastroenteritis. "
              "Take paracetamol 500 mg as needed and dioralyte.")
    print(json.dumps(extract_entities(sample), indent=2))
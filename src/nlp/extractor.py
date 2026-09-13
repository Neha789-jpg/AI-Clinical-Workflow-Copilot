import re

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


def extract_entities(transcript):
    low = transcript.lower()

    symptoms = [{"text": s, "duration": find_duration(transcript, s)}
                for s in find_terms(transcript, SYMPTOM_VOCAB)]

    history = [h for h in find_terms(transcript, HISTORY_VOCAB)
               if re.search(r"(history|previous|past|diagnosed with|known|have had|i have)\W{0,60}" + re.escape(h), low)]

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

    history = [h for h in find_terms(transcript, HISTORY_VOCAB)
               if re.search(r"(history|previous|past|diagnosed with|known|have had|i have)\W{0,60}" + re.escape(h), low)]

    return {
        "patient_info": find_patient_info(transcript),
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "medications": medications,
        "allergies": allergies,
        "vitals": find_vitals(transcript),
        "history": history,
    }


if __name__ == "__main__":
    import json
    sample = ("Doctor: How can I help? Patient: I'm a 34 year old woman, I've had really bad "
              "diarrhea for the last 3 days and some stomach pain, and a fever on the first day. "
              "I have asthma and use salbutamol. No known allergies. "
              "Doctor: BP is 120/80, pulse 88, temperature 37.2. I think this is gastroenteritis. "
              "Take paracetamol 500 mg as needed and dioralyte.")
    print(json.dumps(extract_entities(sample), indent=2))
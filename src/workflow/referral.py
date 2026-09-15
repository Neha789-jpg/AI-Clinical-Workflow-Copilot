
# Step 5 of the pipeline - decide if the patient needs a referral,
# pick the department, and write the referral letter.
#
# Two ways to do it (same idea as extractor.py):
#   - LLM version   -> used when GROQ_API_KEY is set in .env
#   - rules version -> keyword matching + template letter, works offline

import os
import json
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------
# Rules version
# ----------------------------------------------------------

# keywords for each department (used to guess where to refer)
SPECIALTY_MAP = {
    "Neurology": ["dizziness", "vertigo", "migraine", "seizure", "numbness", "stroke", "neuro"],
    "Cardiology": ["chest pain", "palpitations", "angina", "heart failure", "cardio"],
    "Orthopaedics": ["knee", "fracture", "joint injury", "back pain", "ligament", "ortho"],
    "Gastroenterology": ["bowel", "rectal bleeding", "ibs", "crohn", "colitis", "gastro"],
    "Dermatology": ["rash", "eczema", "psoriasis", "mole", "skin"],
    "Ophthalmology": ["vision", "eye", "blurred"],
    "ENT": ["hearing", "tinnitus", "sinus", "tonsil"],
    "Dental": ["dental", "tooth", "toothache", "dentist"],
    "Urology": ["haematuria", "kidney stone", "prostate", "urolog"],
}

# words the doctor uses when they are referring someone
REFERRAL_WORDS = ["refer", "referral", "specialist", "see a", "send you to"]


def needs_referral(transcript):
    # did the doctor actually mention referring the patient?
    text = transcript.lower()
    for word in REFERRAL_WORDS:
        if word in text:
            return True
    return False


def pick_specialty(transcript, entities):
    # count how many keywords of each department appear, pick the highest
    text = transcript.lower() + " " + json.dumps(entities).lower()

    best_specialty = "General Medicine"
    best_score = 0

    for specialty, keywords in SPECIALTY_MAP.items():
        score = 0
        for word in keywords:
            score += text.count(word)
        if score > best_score:
            best_score = score
            best_specialty = specialty

    return best_specialty


def get_symptom_text(entities):
    parts = []
    for s in entities.get("symptoms", []):
        if s.get("duration"):
            parts.append(f"{s['text']} for {s['duration']}")
        else:
            parts.append(s["text"])
    if parts:
        return ", ".join(parts)
    return "symptoms as discussed"


def get_history_text(entities):
    history = entities.get("medical_history", []) + entities.get("family_history", [])
    if history:
        return "; ".join(history)
    return "nil significant"


def write_letter(specialty, entities, soap):
    symptoms = get_symptom_text(entities)
    history = get_history_text(entities)
    diagnosis = soap.get("assessment", "").replace("Assessment: ", "")
    if not diagnosis:
        diagnosis = "under evaluation"

    letter = f"Dear {specialty} Department,\n\n"
    letter += f"I would be grateful if you could see this patient, who presents with {symptoms}.\n"
    letter += f"Working diagnosis: {diagnosis}\n"
    letter += f"Relevant history: {history}\n\n"
    letter += "Please evaluate for further diagnosis and management.\n\n"
    letter += "Kind regards,\nGP"
    return letter


def generate_referral_rules(transcript, entities, soap):
    if not needs_referral(transcript):
        return {
            "referral_needed": False,
            "specialist": None,
            "referral_reason": "",
            "referral_letter": "",
        }

    specialty = pick_specialty(transcript, entities)
    return {
        "referral_needed": True,
        "specialist": specialty,
        "referral_reason": soap.get("assessment", ""),
        "referral_letter": write_letter(specialty, entities, soap),
    }


# ----------------------------------------------------------
# LLM version
# ----------------------------------------------------------

PROMPT = """You are a UK GP writing a specialist referral letter for the NHS.
You will get the consultation transcript, the extracted clinical information, and the SOAP note.

Return ONLY a JSON object like this:
{
  "referral_needed": true or false,
  "specialist": "department name (e.g. Neurology, Cardiology, Orthopaedics, Dental, Ophthalmology, ENT, Gastroenterology, Dermatology, Urology)" or null,
  "referral_reason": "one sentence",
  "referral_letter": "the full letter"
}

Rules:
- NEVER name a specific test (MRI, CT, X-ray, blood test) unless the doctor said that exact test.
  If the doctor said "brain scan", write "brain scan". If unsure, write "appropriate investigations".
- referral_needed is true ONLY if the doctor decided to refer the patient in the transcript.
  If there was no referral, set it to false and leave the other fields null or empty.
- The letter should be short and formal, using only facts from the transcript:
  Dear <Department>, presenting complaint and how long, relevant history and medications,
  examination findings if any, working diagnosis, what you want the specialist to do,
  then "Kind regards," and "GP".
- Do not make up test results, names, dates or NHS numbers.
- No markdown and no explanation, just the JSON.
- If the doctor indicated urgency (urgent, two-week wait, same day), state it clearly at the top of the letter."""

def generate_referral_llm(transcript, entities, soap, retries=2):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    user_message = (
        "TRANSCRIPT:\n" + transcript + "\n\n"
        "CLINICAL INFORMATION:\n" + json.dumps(entities, indent=1) + "\n\n"
        "SOAP NOTE:\n" + json.dumps(soap, indent=1)
    )

    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            return {
                "referral_needed": bool(data.get("referral_needed")),
                "specialist": data.get("specialist"),
                "referral_reason": data.get("referral_reason") or "",
                "referral_letter": data.get("referral_letter") or "",
            }
        except Exception as e:
            last_error = e
            print(f"[referral] attempt {attempt + 1} failed, retrying...")

    raise last_error

# ----------------------------------------------------------
# Main function (this is what the rest of the project calls)
# ----------------------------------------------------------

def generate_referral(transcript, entities, soap, engine="auto"):
    # engine = "auto"  -> LLM if we have a key, otherwise rules
    # engine = "llm"   -> force LLM
    # engine = "rules" -> force rules

    if engine == "rules" or (engine == "auto" and not os.getenv("GROQ_API_KEY")):
        return generate_referral_rules(transcript, entities, soap)

    try:
        return generate_referral_llm(transcript, entities, soap)
    except Exception as e:
        print("[referral] LLM failed, using rules:", e)
        return generate_referral_rules(transcript, entities, soap)


if __name__ == "__main__":
    # quick test with the neurology example from the client doc
    test_entities = {
        "symptoms": [
            {"text": "dizziness", "duration": "3 days"},
            {"text": "nausea", "duration": None},
        ],
        "diagnoses": [{"text": "vestibular disorder", "status": "suspected"}],
        "medical_history": ["previous neurological treatment in India"],
        "family_history": [],
    }
    test_soap = {"assessment": "Assessment: Possible vestibular disorder."}
    test_transcript = (
        "Doctor: You have had dizziness and nausea for three days, and you had "
        "neurological treatment in India before. I think we should refer you to a "
        "neurologist for further tests."
    )

    print("--- RULES ---")
    print(json.dumps(generate_referral(test_transcript, test_entities, test_soap, engine="rules"), indent=2))

    print("\n--- LLM ---")
    print(json.dumps(generate_referral(test_transcript, test_entities, test_soap, engine="llm"), indent=2))
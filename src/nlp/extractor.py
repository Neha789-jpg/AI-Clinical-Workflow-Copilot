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

Your task is to extract ALL clinically relevant information explicitly stated in a doctor-patient consultation transcript.

Return ONLY a valid JSON object with exactly these keys:

{
  "patient_info": {"age": null, "gender": null},

  "symptoms": [
    {
      "text": "...",
      "duration": null,
      "severity": null,
      "status": "current"
    }
  ],
  
  "relevant_context": [
    "..."
  ],

  "diagnoses": [
    {
      "text": "...",
      "status": "confirmed|suspected"
    }
  ],

  "current_medications": [
    {
      "name": "...",
      "dose": null,
      "frequency": null
    }
  ],

  "prescribed": [
    {
      "name": "...",
      "dose": null,
      "frequency": null,
      "duration": null
    }
  ],

  "investigations": ["..."],

  "follow_up": ["..."],

  "advice": ["..."],

  "allergies": ["..."],

  "vitals": {
    "bp": null,
    "pulse": null,
    "temperature": null,
    "spo2": null
  },

  "medical_history": ["..."],
  "family_history": ["..."],
  "social_history": ["..."]
}

GENERAL EXTRACTION RULES:

- Extract information only when it is explicitly stated in the transcript.
- Do not invent, assume, calculate, diagnose, or infer information.
- Preserve the meaning and clinically important details of the original statement.
- Do not replace a specific instruction with a vague summary.
- Do not omit clinically relevant details simply because another field contains related information.
- If several separate instructions are given, preserve them as separate entries when appropriate.
- Do not create duplicate entries.
- Use null for missing single values and [] for missing lists.

SYMPTOMS:

- Include symptoms that the patient currently has or clearly reports having experienced.
- Do NOT include symptoms that the patient explicitly denies.
- Do NOT extract symptoms from questions asked by the doctor.
- Do NOT extract symptoms mentioned only as hypothetical possibilities.
- Use "current" only when the transcript indicates the symptom is ongoing at the time of the consultation.
- Use "resolved" when the transcript explicitly says the symptom has resolved or is no longer present.
- If a symptom occurred earlier in the illness but the transcript does not clearly say whether it is still present, preserve the uncertainty rather than assuming it is current.
- Temporal phrases such as "on the first day", "earlier", "previously", "initially", "last week", or "before this visit" indicate that the symptom occurred in the past and should not automatically be marked as current.
- Do not guess symptom status.
- Preserve clinically relevant details in the symptom text.
- Include duration only when explicitly stated.
- Include severity only when explicitly stated.
- Preserve pain severity such as "8/10", "severe", "mild", etc.
- Preserve clinically important progression such as worsening, improving, intermittent, recurrent, or constant when explicitly stated.

Examples:
"No vomiting" → do NOT include vomiting.
"Denies fever" → do NOT include fever.
"Headache for three days" → include headache with duration "three days".
"Severe chest pain, 8/10" → include the severity exactly as stated.
"Had diarrhea initially but it has now resolved" → include diarrhea with status "resolved".

RELEVANT CONTEXT:

Use relevant_context ONLY for clinically important information that is explicitly
stated in the transcript AND does not belong in any other extraction field.

The purpose of this field is to preserve useful clinical details that would
otherwise be lost because they are not symptoms, diagnoses, medications,
investigations, follow_up, advice, allergies, vitals, or medical/family/social
history.

IMPORTANT:
- Do NOT duplicate information already captured in another field.
- If information belongs in another field, put it ONLY in that field and do NOT
  repeat it in relevant_context.
- Do NOT use relevant_context as a general summary of the consultation.
- Do NOT move medical history, family history, allergies, or medications into
  relevant_context.
- Do NOT infer or invent information.
- Only include information explicitly stated in the transcript.
- Prefer specific clinically useful details over vague statements.

Good examples:

Patient: "I can't go to work because I need to use the toilet every ten minutes."
-> relevant_context:
   ["unable to work because of frequent diarrhoea"]

Patient: "I need to stay close to the toilet all the time."
-> relevant_context:
   ["needs to stay close to the toilet because of frequent bowel movements"]

Patient: "I try not to eat a lot because I'm going to the toilet every ten minutes."
-> relevant_context:
   ["reduced food intake because of frequent bowel movements"]

Patient: "I think I don't drink enough water."
-> relevant_context:
   ["patient reports possibly inadequate fluid intake"]

Patient: "When you press on my tummy, it hurts a little bit."
-> relevant_context:
   ["mild abdominal tenderness on palpation"]

Patient: "My old inhaler ran out and I bought a new brand here."
-> relevant_context:
   ["started using a new inhaler brand after the previous inhaler ran out"]

Patient: "Sometimes when I get anxious, I use my inhaler."
-> relevant_context:
   ["anxiety sometimes triggers inhaler use"]

Patient: "The pain gets a little better after I go to the toilet."
-> relevant_context:
   ["abdominal pain is partially relieved after bowel movement"]

Do NOT produce these as relevant_context because they belong elsewhere:

Asthma
-> medical_history

Inhaler
-> current_medications

NKDA / no known allergies
-> allergies

Father had bowel cancer
-> family_history

Gastroenteritis
-> diagnoses

Drink lots of water
-> advice

See GP if symptoms worsen
-> follow_up

Diarrhoea for three days
-> symptoms

Only include relevant_context entries when there is genuinely useful information
that does not fit the other fields.

DIAGNOSES:
Extract ONLY diagnoses that the DOCTOR is actually assessing for the CURRENT PATIENT.

A diagnosis can be:
- CONFIRMED: the doctor states or clearly establishes that the patient has it.
- SUSPECTED: the doctor considers it possible/likely/probable or is actively investigating it as a possible explanation for the patient's current condition.

If the doctor explicitly names a condition as the explanation for the patient's
current symptoms, extract it as a diagnosis even if the doctor uses informal
language such as "this is normally called..." or "this is usually called...".

Example:

Doctor: "This is normally called gastroenteritis."
-> diagnoses: [
     {"text": "gastroenteritis", "status": "confirmed"}
   ]

Do not omit a diagnosis simply because the doctor does not use the words
"diagnosis" or "diagnosed".

IMPORTANT DISTINCTIONS:

1. CURRENT PATIENT ONLY
- Do NOT extract conditions belonging to the patient's father, mother, sibling, partner, or any other person.
- Family conditions belong ONLY in family_history.

2. PAST MEDICAL HISTORY IS NOT A CURRENT DIAGNOSIS
- Do NOT extract a condition merely because the patient had it in the past.
- Previous illnesses, previous diagnoses, old surgeries, and old medical conditions belong in medical_history.
- Example:
  Patient: "I had underactive thyroid a few years ago."
  Doctor: "Yes, you had an underactive thyroid."
  -> Do NOT put underactive thyroid in current diagnoses unless the doctor is assessing it as a current problem.
  -> Put it in medical_history.

3. DO NOT TURN SYMPTOMS INTO DIAGNOSES
- Symptoms such as headache, dizziness, chest pain, diarrhea, numbness, anxiety, rash, fatigue, etc. are NOT diagnoses unless the doctor explicitly diagnoses them as a condition.

4. DIFFERENTIAL DIAGNOSES
- If the doctor genuinely considers a condition as a possible explanation for the patient's current symptoms, include it as SUSPECTED.
- Example:
  "This could be migraine."
  -> migraine, suspected
- Example:
  "One possibility is labyrinthitis."
  -> labyrinthitis, suspected

5. EXAMPLES / GENERAL INFORMATION
Do NOT extract a condition when the doctor only mentions it as:
- a general medical example
- something that can cause a symptom
- background information
- an explanation of a disease
- a hypothetical possibility unrelated to the patient's actual assessment

6. EXPLICITLY RULED OUT
Do NOT extract a diagnosis if the doctor says the patient does NOT have it, rules it out, or explicitly says they are not suggesting it.

Example:
"I'm not suggesting that you have multiple sclerosis."
-> Do NOT extract multiple sclerosis.

7. TESTS DO NOT AUTOMATICALLY MEAN DIAGNOSIS
Do NOT extract a disease merely because:
- a test is ordered for it
- the doctor wants to rule it out
- the doctor mentions it as a reason for testing

Only include it if the doctor is actually considering it for this patient.

8. PRIORITIZE THE DOCTOR'S ASSESSMENT
When patient history, family history, symptoms, and possible diagnoses are all mentioned, prioritize what the DOCTOR concludes or actively assesses about the patient's CURRENT condition.

9. DO NOT HALLUCINATE
Never add a diagnosis that is not supported by the doctor's statements.
Do not infer diagnoses from symptoms alone.
Do not infer diagnoses from medications, tests, or family history alone.

10. STATUS
Use:
- "confirmed" when the doctor establishes/states the diagnosis as present.
- "suspected" when the doctor considers it possible/probable/likely or is investigating it as a current possibility.

If the doctor explicitly says they are NOT suggesting the patient has a condition, exclude it entirely.

EXAMPLES:

Example 1:
Patient: "My father has hypertension."
Doctor: "Okay."
-> diagnoses: []
-> family_history: hypertension

Example 2:
Patient: "I had asthma as a child."
Doctor: "You had asthma previously."
-> diagnoses: []
-> medical_history: asthma

Example 3:
Patient: "I've had dizziness and ringing in my ears."
Doctor: "The most common diagnosis for this is labyrinthitis."
-> diagnoses: [
     {"text": "labyrinthitis", "status": "suspected"}
   ]

Example 4:
Doctor: "Sometimes this can be multiple sclerosis, but I'm not suggesting that you have it."
-> diagnoses: []

Example 5:
Doctor: "I'm concerned this could be an anaphylactic reaction because you're having breathing difficulties."
-> diagnoses: [
     {"text": "anaphylactic reaction", "status": "suspected"}
   ]

Example 6:
Patient: "I have diarrhea."
Doctor: "You may have gastroenteritis."
-> diagnoses: [
     {"text": "gastroenteritis", "status": "suspected"}
   ]

Example 7:
Doctor: "We will order a test to rule out diabetes."
-> diagnoses: []

MEDICATIONS:

current_medications = medicines the patient was already taking before or at the beginning of this consultation.

prescribed = medicines the doctor explicitly starts, prescribes, changes, increases, decreases, continues, or confirms as part of the treatment plan during this consultation.

- Keep each medicine in ONLY ONE of these two lists.
- Extract the medication name exactly as stated.
- Preserve dose, frequency, and duration exactly as stated.
- Never calculate or guess a dose.
- Do not silently change units.
- If the doctor changes a medication dose, record the FINAL instruction.
- Do not lose medication changes.
- If a medication is mentioned only as a general example, option, or possibility, do not place it in prescribed.
- If a medication is recommended as something the patient may take but is not explicitly prescribed, place the instruction in advice when appropriate.
- Do not put medications into investigations or diagnoses.

INVESTIGATIONS:

- Record every test, scan, examination, monitoring procedure, or investigation explicitly recommended, ordered, or requested by the doctor.
- Examples include blood tests, urine tests, stool tests, pregnancy tests, ECGs, echocardiograms, X-rays, CT scans, MRI scans, and other investigations.
- Preserve important specificity, such as the name of the test or the reason for the test, when explicitly stated.
- A test being ordered to rule out a condition belongs in investigations, NOT diagnoses.
- Do not omit an investigation merely because the doctor also gives advice about it.

FOLLOW-UP:

Follow-up means a FUTURE clinical review, reassessment, or contact instruction.

Include statements such as:
- return in a few days
- come back next week
- follow up with the doctor
- attend a review appointment
- return after test results
- contact the doctor for review
- return sooner if symptoms worsen
- seek medical review if symptoms do not improve

- Preserve the actual condition and timing of the follow-up instruction.
- Do not turn follow-up instructions into vague summaries.
- If a follow-up instruction contains multiple clinically important conditions, preserve them.
- A warning that specifically tells the patient to seek medical review belongs in follow_up.

ADVICE:

Advice means instructions about what the patient should DO or AVOID as part of self-care, lifestyle, monitoring, or day-to-day management.

Examples:
- rest
- stay hydrated
- drink plenty of fluids
- avoid alcohol
- avoid certain foods
- keep a symptom diary
- avoid scratching
- take time off work
- monitor symptoms
- use an emollient
- use a prescribed treatment as instructed

- Preserve concrete instructions exactly and do not replace them with vague summaries.
- If the doctor gives several pieces of advice, preserve each clinically meaningful instruction.
- Do not omit advice simply because it is related to a medication or investigation.
- Medication prescriptions themselves belong in prescribed, not advice.
- Investigations themselves belong in investigations, not advice.
- Future medical review belongs in follow_up, not advice.

IMPORTANT DISTINCTION:

If the doctor says:

"Rest, drink plenty of fluids, and come back next week if you are not better."

Extract:

advice:
[
  "rest",
  "drink plenty of fluids"
]

follow_up:
[
  "come back next week if symptoms do not improve"
]

If the doctor says:

"Monitor your fever and call the doctor if it gets worse."

Extract:

advice:
[
  "monitor your fever"
]

follow_up:
[
  "call the doctor if the fever gets worse"
]

Do NOT collapse these into one vague statement.

ALLERGIES:

- If the patient explicitly denies allergies, use ["none known"].
- If the patient explicitly names an allergy, record the specific allergen.
- If the patient says they have an allergy but does not name it, use ["unspecified allergy"].
- Do not infer allergies.

HISTORY:

medical_history:
- Past illnesses, chronic conditions, previous diagnoses, operations, or previous similar episodes explicitly stated.

family_history:
- Medical conditions explicitly stated in relatives or family members.

social_history:
- Smoking, alcohol, occupation, living situation, sexual health, or other relevant social information explicitly stated.

Do not confuse the patient's current illness with past medical history.

VITALS:

Extract only explicitly stated values for:
- blood pressure
- pulse / heart rate
- temperature
- oxygen saturation / SpO2

Return:
- "bp" as a string such as "120/80"
- "pulse" as a number when explicitly stated
- "temperature" as a number when explicitly stated
- "spo2" as a number when explicitly stated

Do not add units to the numeric values.
Do not calculate or infer vital signs.

NEGATION AND UNCERTAINTY:

- Carefully distinguish positive statements from negative statements.
- "No fever" → do not extract fever.
- "Denies chest pain" → do not extract chest pain.
- "Possible pneumonia" → diagnosis: pneumonia, status: suspected, ONLY if the doctor is the one expressing that possibility.
- "Could this be pneumonia?" asked by the patient → do not automatically record pneumonia as a diagnosis.
- Do not treat unclear speech-recognition output as confirmed information.

FINAL QUALITY CHECK:

Before returning the JSON, verify that:

1. Every explicitly stated clinically relevant symptom is captured.
2. Important symptom duration, severity, progression, and resolution are preserved.
3. Every doctor-stated diagnosis is captured with the correct certainty.
4. Current medications and newly prescribed medications are separated correctly.
5. Medication dose/frequency/duration and medication changes are not lost.
6. Every explicitly ordered or recommended investigation is captured.
7. Every concrete piece of patient advice is captured.
8. Every future review or medical-contact instruction is captured under follow_up.
9. Warning signs and conditions for seeking medical review are preserved.
10. No information has been invented or inferred.
11. No denied symptom has been incorrectly extracted.
12. No clinically important instruction has been replaced by a vague summary.

Return ONLY the JSON object. No markdown. No explanation.
"""



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
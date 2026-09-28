"""
Run the pipeline over every consultation and save two tables:
  data/processed/02_Extracted_Entities.csv       one row per consultation, all entities
  data/processed/07_Analytics_Encounter_Data.csv Dataset 07 from the plan - feeds the analytics page

usage:
  python3 -m src.nlp.run_extraction --source public_primock57          # 57 real UK consultations
  python3 -m src.nlp.run_extraction --source public_primock57 --limit 5
  python3 -m src.nlp.run_extraction --engine rules                      # offline, fast, lower quality
"""
import argparse
import json
import time

import pandas as pd

from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap
from src.workflow.referral import generate_referral
from src.workflow.coding import suggest_codes

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
parser.add_argument("--engine", default="auto", choices=["auto", "llm", "rules"])
parser.add_argument("--source", default=None, help="e.g. public_primock57")
parser.add_argument("--sleep", type=int, default=8, help="seconds between LLM calls")
args = parser.parse_args()

df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
if args.source:
    df = df[df.source_type == args.source]
if args.limit:
    df = df.head(args.limit)


def names(items):
    return "; ".join(i.get("name") or i.get("text") or "" for i in items if isinstance(i, dict))


def with_backoff(fn, *a, **kw):
    """Groq free tier returns 429 when we go too fast - wait and retry."""
    for wait in (0, 30, 60):
        try:
            time.sleep(wait)
            return fn(*a, **kw)
        except Exception as e:
            if "429" not in str(e):
                raise
            print("   rate limited, waiting...")
    return fn(*a, **kw)


rows02, rows07 = [], []
for n, r in enumerate(df.itertuples(), 1):
    transcript = str(r.transcript)
    ent = with_backoff(extract_entities, transcript, engine=args.engine)
    soap = generate_soap(transcript, ent)
    referral = with_backoff(generate_referral, transcript, ent, soap, engine=args.engine)
    codes = suggest_codes(ent, engine="rules")

    diagnoses = ent.get("diagnoses", [])
    primary = diagnoses[0]["text"] if diagnoses else ""
    follow_up = ent.get("follow_up", [])

    rows02.append({
        "consultation_id": r.consultation_id,
        "source_type": r.source_type,
        "age": ent.get("patient_info", {}).get("age"),
        "gender": ent.get("patient_info", {}).get("gender"),
        "symptoms": names(ent.get("symptoms", [])),
        "diagnoses": "; ".join(f'{d["text"]} ({d.get("status")})' for d in diagnoses),
        "current_medications": names(ent.get("current_medications", [])),
        "prescribed": names(ent.get("prescribed", [])),
        "allergies": "; ".join(ent.get("allergies", [])),
        "vitals": json.dumps(ent.get("vitals", {})),
        "medical_history": "; ".join(ent.get("medical_history", [])),
        "family_history": "; ".join(ent.get("family_history", [])),
        "social_history": "; ".join(ent.get("social_history", [])),
        "investigations": "; ".join(ent.get("investigations", [])),
        "follow_up": "; ".join(follow_up),
        "referral_needed": referral["referral_needed"],
        "specialist": referral["specialist"],
        "snomed_codes": "; ".join(c["snomed_code"] for c in codes if c["snomed_code"]),
        "entities_json": json.dumps(ent),
        "engine": args.engine,
    })

    rows07.append({
        "consultation_id": r.consultation_id,
        "patient_id": r.patient_id,
        "date": r.date,
        "age": ent.get("patient_info", {}).get("age"),
        "gender": ent.get("patient_info", {}).get("gender"),
        "specialty": r.specialty if isinstance(r.specialty, str) and r.specialty else "General Practice",
        "symptoms": names(ent.get("symptoms", [])),
        "diagnosis": primary,
        "medications": names(ent.get("current_medications", []) + ent.get("prescribed", [])),
        "referral": bool(referral["referral_needed"]),
        "followup_required": bool(follow_up),
        "source_type": r.source_type,
    })

    print(f"{n}/{len(df)}  {r.consultation_id}  {primary[:40]:<40} referral={referral['referral_needed']}")
    if args.engine != "rules":
        time.sleep(args.sleep)

    pd.DataFrame(rows02).to_csv("data/processed/02_Extracted_Entities.csv", index=False)
    pd.DataFrame(rows07).to_csv("data/processed/07_Analytics_Encounter_Data.csv", index=False)

print(f"\nSaved {len(rows02)} rows to data/processed/02_Extracted_Entities.csv and 07_Analytics_Encounter_Data.csv")

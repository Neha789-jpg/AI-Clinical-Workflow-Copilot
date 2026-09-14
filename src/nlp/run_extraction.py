"""
Run entity extraction over every consultation and save the results.

usage:
  python3 -m src.nlp.run_extraction                 # all rows, LLM (slow, uses API)
  python3 -m src.nlp.run_extraction --limit 20      # first 20 only (good for testing)
  python3 -m src.nlp.run_extraction --engine rules  # offline, fast, lower quality
"""
import argparse
import json
import time
import pandas as pd
from src.nlp.extractor import extract_entities

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
parser.add_argument("--engine", default="auto", choices=["auto", "llm", "rules"])
parser.add_argument("--source", default=None, help="e.g. public_primock57 to filter one dataset")
args = parser.parse_args()

df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
if args.source:
    df = df[df.source_type == args.source]
if args.limit:
    df = df.head(args.limit)

rows = []
for i, r in df.iterrows():
    ent = extract_entities(str(r.transcript), engine=args.engine)
    rows.append({
        "consultation_id": r.consultation_id,
        "source_type": r.source_type,
        "age": ent["patient_info"].get("age"),
        "gender": ent["patient_info"].get("gender"),
        "symptoms": "; ".join(s["text"] for s in ent["symptoms"]),
        "diagnoses": "; ".join(f'{d["text"]} ({d["status"]})' for d in ent["diagnoses"]),
        "medications": "; ".join(m["name"] for m in ent["medications"]),
        "allergies": "; ".join(ent["allergies"]),
        "vitals": json.dumps(ent["vitals"]),
        "history": "; ".join(ent["history"]),
        "entities_json": json.dumps(ent),
        "engine": args.engine,
    })
    print(f"{len(rows)}/{len(df)}  {r.consultation_id}  {rows[-1]['diagnoses'][:60]}")
    if args.engine != "rules":
        time.sleep(1)   # be polite to the free API tier

out = pd.DataFrame(rows)
out.to_csv("data/processed/02_Extracted_Entities.csv", index=False)
out.to_excel("data/processed/02_Extracted_Entities.xlsx", index=False)
print(f"\nSaved {len(out)} rows to data/processed/02_Extracted_Entities.csv")
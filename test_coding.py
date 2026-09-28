import sys
import json
import pandas as pd
from src.nlp.extractor import extract_entities
from src.workflow.coding import suggest_codes

# usage: python3 test_coding.py C00003
cid = sys.argv[1] if len(sys.argv) > 1 else "C00001"
df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
row = df[df.consultation_id == cid].iloc[0]

entities = extract_entities(row.transcript)
print(f"=== {cid} | complaint: {row.symptoms}\n")
print("Diagnoses found:", [d["text"] for d in entities.get("diagnoses", [])])
print("\nSNOMED suggestions:")
print(json.dumps(suggest_codes(entities), indent=2))

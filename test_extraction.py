import sys
import json
import pandas as pd
from src.nlp.extractor import extract_entities

# usage: python3 test_extraction.py C00005
cid = sys.argv[1] if len(sys.argv) > 1 else "C00001"

consults = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
notes = pd.read_excel("data/processed/03_SOAP_Clinical_Documentation.xlsx")

c = consults[consults.consultation_id == cid].iloc[0]
n = notes[notes.consultation_id == cid].iloc[0]

print(f"=== {cid} | source: {c.source_type} | complaint: {c.symptoms}\n")
print("--- EXTRACTED BY OUR MODEL ---")
print(json.dumps(extract_entities(c.transcript), indent=2))
print("\n--- WHAT THE REAL DOCTOR WROTE ---")
print("Subjective:", n.subjective)
print("Assessment:", n.assessment)
print("Plan:", n.plan)

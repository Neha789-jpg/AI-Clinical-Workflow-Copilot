import sys
import json
import pandas as pd
from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap
from src.workflow.referral import generate_referral

cid = sys.argv[1] if len(sys.argv) > 1 else "C00001"

df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
row = df[df.consultation_id == cid].iloc[0]

entities = extract_entities(row.transcript)
soap = generate_soap(row.transcript, entities)
referral = generate_referral(row.transcript, entities, soap)

print(f"=== {cid} | complaint: {row.symptoms}\n")
print("Referral needed:", referral["referral_needed"])
print("Specialist:", referral["specialist"])
print("Reason:", referral["referral_reason"])
print("\n--- LETTER ---")
print(referral["referral_letter"])
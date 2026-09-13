import json
import pandas as pd
from src.nlp.extractor import extract_entities

df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")
row = df[df.source_type == "public_primock57"].iloc[0]

print("Consultation:", row.consultation_id, "| complaint:", row.symptoms)
print(json.dumps(extract_entities(row.transcript), indent=2))
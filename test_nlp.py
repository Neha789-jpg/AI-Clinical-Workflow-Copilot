import json
import pandas as pd
from src.nlp.extractor import extract_entities

df = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx")

# Test first 5 consultations
for i in range(5):
    row = df.iloc[i]

    print("\n" + "=" * 60)
    print("CONSULTATION:", row["consultation_id"])
    print("=" * 60)

    print("\n--- TRANSCRIPT ---")
    print(row["transcript"])

    print("\n--- EXTRACTED INFORMATION ---")
    result = extract_entities(row["transcript"])
    print(json.dumps(result, indent=2))
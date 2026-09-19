import pandas as pd
import json

from src.nlp.extractor import extract_entities


# 1. Load the written consultation dataset
file_path = "data/processed/01_Consultation_Transcription.xlsx"

df = pd.read_excel(file_path)


# 2. Select C00001
consultation = df[df["consultation_id"] == "C00001"].iloc[0]

transcript = consultation["transcript"]


# 3. Run our NLP extractor
model_output = extract_entities(transcript, engine="llm")


# 4. Display reference information
print("\n===== REFERENCE INFORMATION =====")

for column in [
    "symptoms",
    "medical_history",
    "allergies",
    "vital_signs"
]:
    print(f"\n{column}:")
    print(consultation[column])


# 5. Display model output
print("\n===== MODEL OUTPUT =====")

print(json.dumps(model_output, indent=4, ensure_ascii=False))
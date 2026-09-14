import json
import pandas as pd

from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap


# Load the SOAP dataset
df = pd.read_excel("data/processed/03_SOAP_Clinical_Documentation.xlsx")


# Test first 5 consultations
for i in range(5):

    row = df.iloc[i]

    transcript = row["transcript"]

    # Extract clinical information using the NLP module
    entities = extract_entities(transcript)

    # Generate SOAP note using our documentation module
    soap = generate_soap(transcript, entities)

    print("\n" + "=" * 70)
    print("CONSULTATION:", row["consultation_id"])
    print("=" * 70)

    print("\n--- GENERATED SUBJECTIVE ---")
    print(soap["subjective"])

    print("\n--- REFERENCE SUBJECTIVE ---")
    print(row["subjective"])

    print("\n--- GENERATED SOAP ---")
    print(json.dumps(soap, indent=2))
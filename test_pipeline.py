from src.transcription.transcriber import transcribe_audio
from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap


audio_path = "data/primock57/extracted/day5_consultation04.wav"

# Step 1: Audio → Transcript
transcript = transcribe_audio(audio_path)

print("\n--- TRANSCRIPT ---")
print(transcript)

# Step 2: Transcript → Clinical Information
entities = extract_entities(transcript)

print("\n--- EXTRACTED CLINICAL INFORMATION ---")
import json

print("\n--- EXTRACTED CLINICAL INFORMATION ---")
print(json.dumps(entities, indent=2))

# Step 3: Clinical Information → SOAP Documentation
soap_note = generate_soap(transcript, entities)

print("\n--- GENERATED SOAP NOTE ---")

for section, content in soap_note.items():
    print(f"\n{section.upper()}")
    print(content)
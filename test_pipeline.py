from src.transcription.transcriber import transcribe_audio
from src.nlp.extractor import extract_entities


audio_path = "data/consulation.mpeg"

# Step 1: Audio → Transcript
transcript = transcribe_audio(audio_path)

print("\n--- TRANSCRIPT ---")
print(transcript)

# Step 2: Transcript → Clinical Information
entities = extract_entities(transcript)

print("\n--- EXTRACTED CLINICAL INFORMATION ---")
print(entities)
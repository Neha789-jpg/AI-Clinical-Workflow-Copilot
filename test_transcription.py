from src.transcription.transcriber import transcribe_audio

audio_path = "data/consulation.mpeg"

text = transcribe_audio(audio_path)

print("\nTranscription:")
print(text)
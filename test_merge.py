from src.transcription.audio_utils import merge_audio
from src.transcription.transcriber import transcribe_audio

merged = merge_audio(
    "data/raw/primock57/audio/day3_consultation09_doctor.wav",
    "data/raw/primock57/audio/day3_consultation09_patient.wav",
)
print("merged file:", merged)
print(transcribe_audio(merged)[:1500])

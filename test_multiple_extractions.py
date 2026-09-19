from pathlib import Path
from src.transcription.transcriber import transcribe_audio

# Folder containing extracted audio files
audio_folder = Path("data/primock57/extracted")

# Get all WAV files
audio_files = sorted(audio_folder.glob("*.wav"))

# Transcribe each audio file
for audio_path in audio_files:

    print("\n" + "=" * 60)
    print(f"FILE: {audio_path.name}")
    print("=" * 60)

    transcript = transcribe_audio(str(audio_path))

    print("\nTRANSCRIPT:")
    print(transcript)
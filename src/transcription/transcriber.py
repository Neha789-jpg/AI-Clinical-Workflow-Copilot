import whisper

# Load the model once when this file is imported
model = whisper.load_model("small.en")


def transcribe_audio(audio_path):
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False
    )

    return result["text"]
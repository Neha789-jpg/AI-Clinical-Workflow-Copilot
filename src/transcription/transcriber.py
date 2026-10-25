import os

_local_model = None


def _transcribe_groq(audio_path):
    """Hosted Whisper (large-v3-turbo) via Groq — used in deployment."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    with open(audio_path, "rb") as f:
        r = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo", file=f, language="en", temperature=0
        )
    return r.text


def _transcribe_local(audio_path):
    """Local Whisper small.en — used for the paper's evaluation."""
    global _local_model
    import whisper  # imported lazily so the app runs without it installed
    if _local_model is None:
        _local_model = whisper.load_model("small.en")
    result = _local_model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
    )
    return result["text"]


def transcribe_audio(audio_path):
    """
    USE_GROQ_WHISPER=1 (default) and a GROQ_API_KEY -> hosted Whisper.
    USE_GROQ_WHISPER=0, or no key, or hosted call fails -> local small.en.
    """
    if os.getenv("GROQ_API_KEY") and os.getenv("USE_GROQ_WHISPER", "1") == "1":
        try:
            return _transcribe_groq(audio_path)
        except Exception as e:
            print(f"[transcriber] hosted Whisper failed ({e}); falling back to local small.en")
    return _transcribe_local(audio_path)
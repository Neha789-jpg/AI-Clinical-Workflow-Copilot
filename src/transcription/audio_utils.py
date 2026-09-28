# audio_utils.py
# PriMock57 records the doctor and patient on separate microphones.
# Whisper needs one file, so this mixes the two tracks together using ffmpeg.

import subprocess
import tempfile


def merge_audio(doctor_path, patient_path):
    """Mix two audio files into one and return the path of the merged file."""
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    command = [
        "ffmpeg", "-y",
        "-i", doctor_path,
        "-i", patient_path,
        "-filter_complex", "amix=inputs=2:duration=longest",
        out,
    ]
    subprocess.run(command, check=True, capture_output=True)
    return out
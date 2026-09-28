import os
import re
import pandas as pd
from jiwer import wer, cer

from src.transcription.transcriber import transcribe_audio


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "01_Consultation_Transcription.xlsx"
)

AUDIO_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "primock57",
    "extracted"
)

RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_reference(text):
    """
    Cleans the reference transcript before comparison.

    We remove:
    - Doctor:/Patient: speaker labels
    - extra whitespace
    - line breaks

    We keep the actual spoken words because WER should
    measure transcription accuracy.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove speaker labels
    text = re.sub(r"\b(?:Doctor|Patient)\s*:\s*", " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_whisper(text):
    """
    Cleans Whisper's output so that comparison is consistent.
    """

    if not text:
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# FIND REFERENCE TRANSCRIPT
# --------------------------------------------------

def load_reference_dataset():
    """
    Loads the consultation dataset containing the
    ground-truth/reference transcripts.
    """

    df = pd.read_excel(DATASET_PATH)

    required_columns = {
        "consultation_id",
        "source_id",
        "transcript"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return df


# --------------------------------------------------
# FIND AVAILABLE AUDIO FILES
# --------------------------------------------------

def find_audio_files():
    """
    Finds all WAV files that are actually present
    in data/primock57/extracted.
    """

    if not os.path.exists(AUDIO_FOLDER):
        raise FileNotFoundError(
            f"Audio folder not found:\n{AUDIO_FOLDER}"
        )

    audio_files = {}

    for filename in os.listdir(AUDIO_FOLDER):

        if filename.lower().endswith(".wav"):

            # Example:
            # day5_consultation01.wav
            source_id = os.path.splitext(filename)[0]

            audio_files[source_id] = os.path.join(
                AUDIO_FOLDER,
                filename
            )

    return audio_files


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TRANSCRIPTION EVALUATION")
    print("=" * 60)

    print("\nLoading reference dataset...")
    df = load_reference_dataset()

    print(f"Dataset contains {len(df)} consultations.")

    print("\nFinding available audio files...")
    audio_files = find_audio_files()

    print(f"Found {len(audio_files)} local audio files.")

    results = []

    for source_id, audio_path in audio_files.items():

        print("\n" + "-" * 60)
        print(f"Evaluating: {source_id}")
        print("-" * 60)

        # Find corresponding reference row
        matches = df[df["source_id"] == source_id]

        if matches.empty:
            print("WARNING: No reference transcript found. Skipping.")
            continue

        row = matches.iloc[0]

        consultation_id = row["consultation_id"]

        reference = normalize_reference(
            row["transcript"]
        )

        print(f"Consultation ID: {consultation_id}")
        print("Running Whisper...")

        try:

            hypothesis = transcribe_audio(audio_path)

            hypothesis = normalize_whisper(hypothesis)

            # Calculate metrics
            word_error_rate = wer(
                reference,
                hypothesis
            )

            character_error_rate = cer(
                reference,
                hypothesis
            )

            print(f"WER: {word_error_rate:.4f}")
            print(f"CER: {character_error_rate:.4f}")

            results.append({
                "consultation_id": consultation_id,
                "source_id": source_id,
                "audio_path": audio_path,
                "reference_transcript": reference,
                "whisper_transcript": hypothesis,
                "wer": word_error_rate,
                "cer": character_error_rate
            })

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "consultation_id": consultation_id,
                "source_id": source_id,
                "audio_path": audio_path,
                "reference_transcript": reference,
                "whisper_transcript": "",
                "wer": None,
                "cer": None
            })


    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    if not results:
        print("\nNo consultations were successfully evaluated.")
        return

    results_df = pd.DataFrame(results)

    output_path = os.path.join(
        RESULTS_FOLDER,
        "transcription_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # Overall metrics
    valid_results = results_df.dropna(
        subset=["wer", "cer"]
    )

    print("\n" + "=" * 60)
    print("TRANSCRIPTION EVALUATION RESULTS")
    print("=" * 60)

    print(f"Consultations evaluated: {len(valid_results)}")

    print(
        f"Average WER: "
        f"{valid_results['wer'].mean():.4f}"
    )

    print(
        f"Average CER: "
        f"{valid_results['cer'].mean():.4f}"
    )

    print("\nResults saved to:")
    print(output_path)

    print("=" * 60)


if __name__ == "__main__":
    main()
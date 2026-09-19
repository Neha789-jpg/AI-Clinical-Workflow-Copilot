import pandas as pd
from pathlib import Path

# Dataset location
file_path = "data/primock57/train-00000-of-00002.parquet"

# Read the dataset
df = pd.read_parquet(file_path)

# Create output folder
output_folder = Path("data/primock57/extracted")
output_folder.mkdir(parents=True, exist_ok=True)

# Select the first 5 consultations
for index, row in df.head(5).iterrows():

    # Extract audio information
    audio = row["audio"]

    # Get the audio bytes
    audio_bytes = audio["bytes"]

    # Create output file path
    output_path = output_folder / row["file_name"]

    # Skip files that already exist
    if output_path.exists():
        print(f"Already exists: {output_path.name}")
        continue

    # Save audio as WAV
    with open(output_path, "wb") as file:
        file.write(audio_bytes)

    print(f"Audio extracted: {output_path.name}")
    print(f"File size: {len(audio_bytes)} bytes")

print("\nExtraction completed!")
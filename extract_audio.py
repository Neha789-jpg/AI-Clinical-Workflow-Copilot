import pandas as pd
from pathlib import Path

# Dataset location
file_path = "data/primock57/train-00000-of-00002.parquet"

# Read the dataset
df = pd.read_parquet(file_path)

# Select the first consultation
row = df.iloc[0]

# Extract audio information
audio = row["audio"]

# Get the audio bytes
audio_bytes = audio["bytes"]

# Create output folder
output_folder = Path("data/primock57/extracted")
output_folder.mkdir(parents=True, exist_ok=True)

# Create the output file path
output_path = output_folder / row["file_name"]

# Save the audio bytes as a WAV file
with open(output_path, "wb") as file:
    file.write(audio_bytes)

print("Audio extracted successfully!")
print("Saved at:", output_path)
print("File size:", len(audio_bytes), "bytes")
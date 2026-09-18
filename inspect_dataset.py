import pandas as pd

file_path = "data/primock57/train-00000-of-00002.parquet"

df = pd.read_parquet(file_path)

# Get the first consultation
row = df.iloc[0]

# Get the audio information
audio = row["audio"]

print("File name:", row["file_name"])
print("\nTranscript preview:")
print(row["transcript"][:300])

print("\nAudio dictionary keys:")
print(audio.keys())

print("\nAudio path:")
print(audio.get("path"))

print("\nAudio data size:")
print(len(audio["bytes"]), "bytes")
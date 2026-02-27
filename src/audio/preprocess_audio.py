import os
import numpy as np
import librosa

DATA_DIR = "data/audio/raw"
OUTPUT_DIR = "data/audio/processed"

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_DURATION = 6  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pad_or_trim(audio, max_len):
    if len(audio) > max_len:
        return audio[:max_len]
    return np.pad(audio, (0, max_len - len(audio)))

def extract_mfcc(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio = pad_or_trim(audio, SAMPLE_RATE * MAX_DURATION)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )
    return mfcc

X, y = [], []

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            path = os.path.join(label_path, file)
            X.append(extract_mfcc(path))
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

np.save(os.path.join(OUTPUT_DIR, "X_mfcc.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

print(" MFCC preprocessing done")
print("X shape:", X.shape)
print("y shape:", y.shape)
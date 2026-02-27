import os
import librosa

DATA_DIR = "data/audio/raw"
SAMPLE_RATE = 16000

def scan_dataset(data_dir=DATA_DIR):
    samples = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.lower().endswith(".wav"):
                samples.append({
                    "path": os.path.join(label_path, file),
                    "label": label
                })
    return samples

def test_load(sample):
    y, sr = librosa.load(sample["path"], sr=SAMPLE_RATE)
    return y, sr

if __name__ == "__main__":
    samples = scan_dataset()
    print(f"Total samples found: {len(samples)}")

    test_sample = samples[0]
    audio, sr = test_load(test_sample)

    print("Label:", test_sample["label"])
    print("Audio shape:", audio.shape)
    print("Sample rate:", sr)
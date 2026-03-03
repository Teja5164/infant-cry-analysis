import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa

# ---------------- CONFIG ----------------
MODEL_PATH = "audio_model.pth"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_DURATION = 6
DEVICE = "cpu"
# ----------------------------------------


# Same model as training
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Load model
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    labels = checkpoint["labels"].classes_
    model = AudioCNN(len(labels))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, labels


model, labels = load_model()


def preprocess_audio(file):
    audio, _ = librosa.load(file, sr=SAMPLE_RATE)
    max_len = SAMPLE_RATE * MAX_DURATION

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ---------------- UI ----------------
st.title("🍼 Infant Cry Analysis System")
st.write("Upload a cry audio file (.wav)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    input_tensor = preprocess_audio(uploaded_file)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction")
    st.write(f"**Cry Type:** {labels[pred]}")
    st.write(f"Confidence: {probs[0][pred].item():.2f}")
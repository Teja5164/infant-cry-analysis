import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Load processed MFCCs
X = np.load("data/audio/processed/X_mfcc.npy")
y = np.load("data/audio/processed/y_labels.npy")

le = LabelEncoder()
y = le.fit_transform(y)

class CryDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

loader = DataLoader(CryDataset(X), batch_size=16, shuffle=False)

# CNN model (same as training, but without classifier)
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AudioCNN().to(device)

# Load trained weights
ckpt = torch.load(
    "audio_model.pth",
    map_location=device,
    weights_only=False   
)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

embeddings = []

with torch.no_grad():
    for xb in loader:
        xb = xb.to(device)
        emb = model(xb)
        embeddings.append(emb.cpu().numpy())
        
embeddings = np.vstack(embeddings)

np.save("data/audio/processed/audio_embeddings.npy", embeddings)
print(" Audio embeddings saved:", embeddings.shape)
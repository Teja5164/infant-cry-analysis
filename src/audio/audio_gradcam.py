import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5          # must match training
MODEL_PATH = "audio_model.pth"
# ----------------------------------------


# ✅ EXACT SAME MODEL AS TRAINING
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # net.0
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),  # net.3
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------- LOAD MODEL ----------------
model = AudioCNN(NUM_CLASSES).to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(ckpt["model"])
model.eval()


# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()


# ---------------- RUN ----------------
X = np.load("data/audio/processed/X_mfcc.npy")

sample = torch.tensor(
    X[0], dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(DEVICE)

gradcam = GradCAM(model, model.net[3])  # 🔥 conv2 layer
cam = gradcam.generate(sample)[0]

plt.imshow(cam, aspect="auto", cmap="jet")
plt.colorbar()
plt.title("Audio Grad-CAM")
plt.show()
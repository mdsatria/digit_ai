# =====================================================
# FLASK + PYTORCH MNIST CNN INFERENCE SERVER
# =====================================================

import base64
import re

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image

# =====================================================
# INIT FLASK
# =====================================================
app = Flask(__name__)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =====================================================
# CNN MODEL DEFINITION
# (Must match training architecture)
# =====================================================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "models/mnist_cnn.pt"

model = MNIST_CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ PyTorch CNN model loaded. Start serving...")


# =====================================================
# BASE64 → PNG
# =====================================================
def convertImage(imgData1):
    imgstr = re.search(b"base64,(.*)", imgData1).group(1)
    with open("output.png", "wb") as output:
        output.write(base64.b64decode(imgstr))


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")


# =====================================================
# PREDICTION ROUTE
# =====================================================
@app.route("/predict/", methods=["POST"])
def predict():

    # Get raw canvas image
    imgData = request.get_data()
    convertImage(imgData)

    # =================================================
    # PREPROCESS (same as CNN training)
    # =================================================
    n_size = 28

    img = Image.open("output.png").resize((n_size, n_size)).convert("L")

    x = np.array(img)

    # Invert colors
    x = np.invert(x)

    # Normalize
    x = x.astype(np.float32) / 255.0

    # Shape → (1, 1, 28, 28)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)

    # =================================================
    # INFERENCE
    # =================================================
    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

    print("Prediction:", predicted_class)
    print("Confidence:", confidence)

    # =================================================
    # RETURN JSON
    # =================================================
    return jsonify(
        {"label": predicted_class, "confidence": confidence, "probs": probs.tolist()}
    )


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)

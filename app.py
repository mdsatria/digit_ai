# =====================================================
# FLASK + SKLEARN MNIST INFERENCE SERVER
# =====================================================

# Flask untuk web interface
import base64
import re

import joblib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

# =====================================================
# INIT FLASK
# =====================================================
app = Flask(__name__)

# =====================================================
# LOAD SKLEARN MODEL
# =====================================================
MODEL_PATH = "models/mnist_mlp.pkl"

model = joblib.load(MODEL_PATH)

print("Sklearn model loaded. Start serving...")


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
@app.route("/predict/", methods=["GET", "POST"])
def predict():

    # Ambil raw image dari canvas
    imgData = request.get_data()
    print("imgData type:", type(imgData))

    # Decode → PNG
    convertImage(imgData)

    # =================================================
    # PREPROCESS (SAMA SEPERTI TRAINING)
    # =================================================
    n_size = 28

    x = Image.open("output.png").resize((n_size, n_size)).convert("L")

    x = np.array(x)

    # Invert warna (canvas putih → hitam)
    x = np.invert(x)

    # Normalisasi
    x = x.astype(np.float32) / 255.0

    # Flatten
    x = x.reshape(1, -1)

    # =================================================
    # SKLEARN PREDICTION
    # =================================================
    probs = model.predict_proba(x)

    predicted_class = np.argmax(probs, axis=1)[0]
    confidence = probs[0][predicted_class]

    print("Probabilities:", probs)
    print("Prediction:", predicted_class, "Confidence:", confidence)

    return str(predicted_class)


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":

    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)

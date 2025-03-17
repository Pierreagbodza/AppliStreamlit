from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load the model once when the application starts
model_path = "best_model.h5"
model = load_model(model_path, compile=False)

@app.route('/', methods=["GET"])
def hello():
    return jsonify({"hello": "Pierre"})

def preprocess(img):
    img = img.resize((224, 224))
    img = np.asarray(img)
    if img.ndim == 2:  # Convert grayscale to RGB
        img = np.stack((img,) * 3, axis=-1)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Récupération de l'image
        file = request.files['file']
        image = file.read()

        # Ouverture de l'image
        img = Image.open(io.BytesIO(image))

        # Traitement de l'image
        img_processed = preprocess(img)

        # Prédiction
        pred = model.predict(img_processed)

        rec = pred[0][0].tolist()

        return jsonify({"predictions": rec})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    

         from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return "🌱 Crop Predictor Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    
    try:
        # Get values in the same order as model was trained
        features = [
            data["N"],
            data["P"],
            data["K"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]
        features = np.array([features])
        prediction = model.predict(features)[0]
        return jsonify({"predicted_crop": prediction})
    
    except KeyError as e:
        return jsonify({"error": f"Missing input field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return "Crop Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N = float(data['nitrogen'])
        P = float(data['phosphorus'])
        K = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(input_data)
        output = prediction[0]

        return jsonify({'prediction': f"Recommended Crop: {output}"})
    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model from the .sav file
try:
    model = joblib.load('heart_disease_model.sav')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    print("Received data:", data)

    # Extract features from the JSON data
    features = [
        data['age'],
        data['sex'],
        data['cp'],
        data['trestbps'],
        data['chol'],
        data['fbs'],
        data['restecg'],
        data['thalach'],
        data['exang'],
        data['oldpeak'],
        data['slope'],
        data['ca'],
        data['thal']
    ]

    # Convert features to numpy array and reshape
    input_data = np.asarray(features).reshape(1, -1)
    print("Input data for prediction:", input_data)

    # Make prediction (get probabilities)
    prediction_proba = model.predict_proba(input_data)
    print("Prediction probabilities:", prediction_proba)

    # Extract the probability of having heart disease (class 1)
    heart_disease_probability = prediction_proba[0][1] * 100  # Convert to percentage

    # Prepare the response
    response = {
        'probability_of_heart_disease': f"{heart_disease_probability:.2f}%",
        'diagnosis': 'The Person is likely to have Heart Disease' if heart_disease_probability >= 50 else 'The Person is likely to be Healthy'
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
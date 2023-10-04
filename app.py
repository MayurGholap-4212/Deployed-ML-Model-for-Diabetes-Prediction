import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib  # Use joblib for model loading

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load the trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the HTML form
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = int(request.form['age'])

        # Prepare input data for prediction
        input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        final_features = [np.array(input_features)]

        # Make a prediction using the loaded model
        prediction = model.predict(final_features)

        # Interpret the prediction (0 or 1) as "Diabetes" or "No Diabetes"
        if prediction[0] == 0:
            output = 'No Diabetes'
        else:
            output = 'Diabetes'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text='An error occurred during prediction.')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get input data as JSON
        data = request.get_json(force=True)

        # Convert JSON data to a list of feature values
        input_features = [int(data['pregnancies']), int(data['glucose']), int(data['blood_pressure']),
                          int(data['skin_thickness']), int(data['insulin']), float(data['bmi']),
                          float(data['diabetes_pedigree_function']), int(data['age'])]

        # Prepare input data for prediction
        final_features = [np.array(input_features)]

        # Make a prediction using the loaded model
        prediction = model.predict(final_features)

        # Interpret the prediction (0 or 1) as "Diabetes" or "No Diabetes"
        if prediction[0] == 0:
            output = 'No Diabetes'
        else:
            output = 'Diabetes'

        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.'})

if __name__ == "__main__":
    app.run(debug=True)

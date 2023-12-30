from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickle models
model = pickle.load(open('model.pkl', 'rb'))
# Assuming encoder is a preprocessor (e.g., scaler or encoder)
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Define the input feature columns
cols = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        input_features = [request.form.get(col) for col in cols]

        # Convert extracted values to float (handling possible None values)
        input_features = [float(value) if value is not None else 0.0 for value in input_features]

        # Create a DataFrame with feature names
        input_df = pd.DataFrame([input_features], columns=cols)

        # Assuming encoder is a preprocessor (e.g., scaler or encoder), use it to transform the input features
        input_features_encoded = encoder.transform(input_df)

        # Make predictions using the loaded model
        prediction = model.predict(input_features_encoded)[0]

        # Print the predicted LC50 value to the console
        print("Predicted LC50 Value:", prediction)

        return render_template('index.html', pred=f'Expected LC50 value will be {prediction:.2f}')
    except Exception as e:
        # Handle any exception and provide an appropriate message
        return render_template('index.html', pred=f'Error predicting LC50 value: {str(e)}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Extract input features from the JSON data
        data = request.get_json(force=True)
        data_unseen = pd.DataFrame([data])

        # Assuming encoder is a preprocessor (e.g., scaler or encoder), use it to transform the input features
        data_unseen_encoded = encoder.transform(data_unseen)

        # Make predictions using the loaded model
        prediction = model.predict(data_unseen_encoded)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        # Handle any exception and provide an appropriate message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)




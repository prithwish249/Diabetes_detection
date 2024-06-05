from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

# Load the trained model, scaler, and feature names
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure there are no unexpected columns
    df = df[feature_names]
    
    # Scale the features
    X_scaled = scaler.transform(df)
    
    # Make predictions
    prediction = model.predict(X_scaled)[0]
    
    # Interpret the prediction
    result = "diabetic" if prediction == 1 else "not diabetic"
    
    # Return the predictions as JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

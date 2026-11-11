import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")

# Load the trained model pipeline
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()
    sample = {
        'Store_Type': data['Store_Type'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Size': data['Store_Size']
    }
    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data).tolist()[0]
    return jsonify({'Sales': prediction})

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)
    predictions = model.predict(input_data).tolist()
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
    return output_dict

if __name__ == '__main__':
    superkart_api.run(debug=True)

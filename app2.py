# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:16:05 2024

@author: bittu
"""

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure model.pkl is in the same directory)
model = joblib.load('carshop.pkl')

# Car name encoding
car_name_mapping = {
    "Audi": 0, "BMW": 1, "Datsun": 2, "Fiat": 3, "Ford": 4, "Honda": 5,
    "Hyundai": 6, "Isuzu": 7, "Jaguar": 8, "Jeep": 9, "Kia": 10, "Land": 11,
    "MG": 12, "Mahindra": 13, "Maruti": 14, "Mercedes-Benz": 15, "Mini": 16,
    "Mitsubishi": 17, "Nexus": 18, "Nissan": 19, "Porsche": 20, "Renault": 21,
    "Skoda": 22, "Tata": 23, "Toyota": 24, "Volkswagen": 25, "Volvo": 26
}

@app.route('/')
def index():
    return render_template('index12.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    car_name = int(request.form['car_name'])
    insurance_validity = int(request.form['insurance_validity'])  # Yes: 1, No: 0
    fuel_type = int(request.form['fuel_type'])  # CNG: 0, Diesel: 1, Petrol: 2
    transmission = int(request.form['transmission'])  # Automatic: 0, Manual: 1
    seats = int(request.form['seats'])
    kms_driven = int(request.form['kms_driven'])
    ownership = int(request.form['ownership'])
    mileage = float(request.form['mileage'])
    engine_cc = float(request.form['engine'])
    car_age = int(request.form['car_age'])

    # Create input array for prediction
    input_data = np.array([[car_name, insurance_validity, fuel_type, seats, kms_driven,
                            ownership, transmission, mileage, engine_cc, car_age]])
    
    # Predict the price
    predicted_price = model.predict(input_data)[0]
    
    # Display the prediction result on the HTML page
    prediction_text = f'Predicted Price: ₹{predicted_price:.2f} Lakhs'
    return render_template('index12.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

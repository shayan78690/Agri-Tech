from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load crop recommendation model and scaler
crop_model = joblib.load("models/crop_model.pkl")
crop_scaler = joblib.load("models/crop_scaler.pkl")

# Load fertilizer recommendation model and scaler
fert_model = joblib.load("models/fert_model.pkl")
fert_scaler = joblib.load("models/fert_scaler.pkl")

# TODO : select from an [api] or fetch it from a [local_db](max no. of params)
# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidney Beans", 5: "Pigeon Peas",
    6: "Moth Beans", 7: "Mung Bean", 8: "Black Gram", 9: "Lentil", 10: "Pomegranate",
    11: "Banana", 12: "Mango", 13: "Grapes", 14: "Watermelon", 15: "Muskmelon",
    16: "Apple", 17: "Orange", 18: "Papaya", 19: "Coconut", 20: "Cotton",
    21: "Jute", 22: "Coffee"
}

# TODO : select from an [api] or fetch it from a [local_db](max no. of params)
# Fertilizer dictionary
fert_dict = {
    1: "Urea", 2: "DAP", 3: "14-35-14", 4: "28-28", 5: "17-17-17",
    6: "20-20", 7: "10-26-26"
}

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Crop Recommendation Page
@app.route('/crop')
def crop():
    return render_template('crop.html')

# Fertilizer Recommendation Page
@app.route('/fert')
def fert():
    return render_template('fert.html')

# Crop Recommendation Result
@app.route('/crop_recommend', methods=['POST'])
def crop_recommend():
    # Get input values
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temp = float(request.form['temp'])
    hum = float(request.form['hum'])
    ph = float(request.form['ph'])
    rain = float(request.form['rain'])

    # Scale features
    features = np.array([[N, P, K, temp, hum, ph, rain]])
    scaled_features = crop_scaler.transform(features)

    # Predict crop
    prediction = crop_model.predict(scaled_features)[0]
    crop_name = crop_dict.get(prediction, "Unknown Crop")

    return render_template('crop_result.html', crop=crop_name)

# Fertilizer Recommendation Result
@app.route('/fert_recommend', methods=['POST'])
def fert_recommend():
    # Get input values
    Temparature = float(request.form['Temparature'])
    Humidity = float(request.form['Humidity'])
    Moisture = float(request.form['Moisture'])
    Soil_Type = int(request.form['Soil_Type'])
    Crop_Type = int(request.form['Crop_Type'])
    Nitrogen = float(request.form['Nitrogen'])
    Potassium = float(request.form['Potassium'])
    Phosphorous = float(request.form['Phosphorous'])

    # Scale features
    features = np.array([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    scaled_features = fert_scaler.transform(features)

    # Predict fertilizer
    prediction = fert_model.predict(scaled_features)[0]
    fertilizer = fert_dict.get(prediction, "Unknown Fertilizer")

    return render_template('fert_result.html', fertilizer=fertilizer)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
    
import numpy as np
import pickle
from flask import Flask, request, render_template, session
import pandas as pd
from sklearn.decomposition import PCA
pca=pickle.load(open('pca.pkl','rb'))

# Load models
best_model_yield = pickle.load(open('best_model_yield.pkl', 'rb'))
best_model_price = pickle.load(open('best_model_price.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
preprocessor_price = pickle.load(open('preprocessor_price.pkl', 'rb'))

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']
        
        # Prepare input for prediction
        features = pd.DataFrame([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]], 
                                columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
        transformed_features = preprocessor.transform(features)
        predicted_value = best_model_yield.predict(transformed_features)[0]  # Extract the scalar prediction
        
        # Store prediction in session
        session['predicted_value'] = predicted_value
        session['Year'] = Year
        session['average_rain_fall_mm_per_year'] = average_rain_fall_mm_per_year
        session['pesticides_tonnes'] = pesticides_tonnes
        session['avg_temp'] = avg_temp
        session['Area'] = Area
        session['Item'] = Item
        return render_template('index.html', prediction=predicted_value, Year=Year, average_rain_fall_mm_per_year=average_rain_fall_mm_per_year, pesticides_tonnes=pesticides_tonnes, avg_temp=avg_temp, Area=Area)
@app.route('/price', methods=['GET', 'POST'])
def price():
    if request.method == 'POST':
        # Retrieve crop yield from session
        crop_yield = session.get('predicted_value', None)
        if crop_yield is None:
            return render_template('price.html', error="Crop yield prediction is missing. Please predict yield first.")

        # Extract other features from the form
        Year = session.get('Year', None)
        average_rain_fall_mm_per_year = session.get('average_rain_fall_mm_per_year', None)
        pesticides_tonnes = session.get('pesticides_tonnes', None)
        avg_temp = session.get('avg_temp', None)
        Area = session.get('Area', None)
        Item = session.get('Item', None)

# Check if any of the required values are missing
        if None in [Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]:
            return render_template('price.html', error="Some required input values are missing. Please provide all inputs.")

        
        # Prepare the input features including Area and Item
        features = pd.DataFrame([[Area, Item, Year, crop_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
                                columns=['Area', 'Item', 'Year', 'crop_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
        
        try:
            # Transform the features using the preprocessor
            transformed_features = preprocessor_price.transform(features)
            print("Transformed features shape:", transformed_features.shape)  # Debugging line to check the shape
            pca_features=pca.transform(transformed_features)
            predicted_price = best_model_price.predict(pca_features)[0]  # Extract the scalar prediction
            return render_template('price.html', price_prediction=predicted_price)
        except Exception as e:
            return render_template('price.html', error=f"Error predicting price: {str(e)}")

    return render_template('price.html')

if __name__ == "__main__":
    app.run(debug=True)

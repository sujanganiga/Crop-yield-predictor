from flask import Flask, request, render_template, send_file, session
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns


# Load models
best_model_yield = pickle.load(open('best_model_yield.pkl', 'rb'))
best_model_price = pickle.load(open('best_model_price.pkl', 'rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))
preprocessor_price = pickle.load(open('preprocessor_price.pkl', 'rb'))
pca=pickle.load(open('pca.pkl','rb'))

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        features = pd.DataFrame([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
                                columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
                                         'avg_temp'])
        transformed_features = preprocessor.transform(features)
        predicted_value = best_model_yield.predict(transformed_features).reshape(1, -1)

        session['predicted_value'] = predicted_value
        session['Year'] = Year
        session['average_rain_fall_mm_per_year'] = average_rain_fall_mm_per_year
        session['pesticides_tonnes'] = pesticides_tonnes
        session['avg_temp'] = avg_temp
        session['Area'] = Area
        session['Item'] = Item
        return render_template('index.html', prediction=predicted_value)


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
            return render_template('price.html',
                                   error="Some required input values are missing. Please provide all inputs.")

        # Prepare the input features including Area and Item
        features = pd.DataFrame(
            [[Area, Item, Year, crop_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
            columns=['Area', 'Item', 'Year', 'crop_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
                     'avg_temp'])

        try:
            # Transform the features using the preprocessor
            transformed_features = preprocessor_price.transform(features)
            print("Transformed features shape:", transformed_features.shape)  # Debugging line to check the shape
            pca_features = pca.transform(transformed_features)
            predicted_price = best_model_price.predict(pca_features)[0]  # Extract the scalar prediction
            return render_template('price.html', price_prediction=predicted_price)
        except Exception as e:
            return render_template('price.html', error=f"Error predicting price: {str(e)}")

    return render_template('price.html')



@app.route("/bulk_predict", methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file_type = request.form.get('value')


    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read the CSV file
    try:
        data = pd.read_csv(file)
    except :
        return "Uploaded file is invalid", 400

    output_file = data

     # Check if the DataFrame is empty
    if data.empty:
        return "Uploaded file has no data. Please upload a populated CSV file.", 400

    if file_type == 'crop':
        # Ensure required columns are present
        required_columns = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        if not all(col in data.columns for col in required_columns):
            return f"Missing required columns: {set(required_columns) - set(data.columns)}", 400
        if len(data.columns) != len(required_columns):
            return f" Extra columns detected: {set(data.columns) - set(required_columns) }", 400


        features = preprocessor.transform(data)
        predictions = best_model_yield.predict(features)

        # Add predictions to the DataFrame
        data['crop_yield'] = predictions

        # Save to CSV
        output_file = 'Crop_predictions.csv'

    elif file_type == 'price':
        # Ensure required columns are present
        required_columns = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp',
                            'crop_yield']
        if not all(col in data.columns for col in required_columns):
            return f"Missing required columns: {set(required_columns) - set(data.columns)}", 400
        if len(data.columns) != len(required_columns):
            return f" Extra columns detected: {set(data.columns) - set(required_columns)}", 400


            # Transform the features using the preprocessor
        transformed_features = preprocessor_price.transform(data)
        pca_features = pca.transform(transformed_features)
        predicted_price = best_model_price.predict(pca_features)  # Extract the scalar prediction

        # Add predictions to the DataFrame
        data['Predicted_price'] = predicted_price

        # Save to CSV
        output_file = 'Price_predictions.csv'

    data.to_csv(output_file, index=False)

    return send_file(output_file, as_attachment=True)



@app.route("/heatmaps", methods=['POST'])
def generate_heatmaps_crops():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_type = request.form.get('value')

    try:
        f = pd.read_csv(file)
    except :
        return "Uploaded file is invalid.", 400

    #check and create validity of the csv
    # Check if the number of rows is less than or equal to 10
    if len(f) <= 10:
        return "Insufficient data: Heatmaps require more than 10 instances.", 400

    # Ensure all required columns are present
    required_columns = [ 'Year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes',
                        'crop_yield', 'Predicted_price']
    for col in required_columns:
        if col not in f.columns:
            f[col] = None  # Add missing columns with default values (e.g., None or 0)

    f.to_csv('updated_file.csv', index=False) #returns none


    # Generate first heatmap: Area, Year, Rain, Temp vs Yield
    if file_type == 'crop':

        f = f.rename(columns={
            'average_rain_fall_mm_per_year': 'Rainfall',
            'avg_temp': 'Temp',
            'crop_yield': 'Yield'
        })
        heatmap1_data = f[[ 'Year', 'Rainfall', 'Temp', 'Yield']].corr()
        plt.figure(figsize=(8, 6))
        plt.title("Correlation Heatmap: Year, Rain, Temp, Yield")
        sns.heatmap(heatmap1_data, annot=True, cmap="coolwarm", fmt=".2f")
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        heatmap_url = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

    elif file_type == 'price':

        f = f.rename(columns={
            'crop_yield': 'Yield',
            'pesticides_tonnes': 'Pesticides',
            'Predicted_price': 'Price'
        })
        # Generate second heatmap: Item, Pesticides, Yield, Price
        heatmap2_data = f[['Pesticides', 'Yield', 'Price']].corr()
        plt.figure(figsize=(8, 6))
        plt.title("Correlation Heatmap:  Pesticides, Yield, Price")
        sns.heatmap(heatmap2_data, annot=True, cmap="coolwarm", fmt=".2f")
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        heatmap_url = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()
    return render_template('heatmaps.html', heatmap_url=heatmap_url)


if __name__ == "__main__":
    app.run(debug=True)

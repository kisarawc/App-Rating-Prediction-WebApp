from flask import Flask, request, render_template
import joblib  # Import joblib to load your model
import numpy as np  # For numerical operations
import pandas as pd  # For DataFrame operations

app = Flask(__name__)

# Load the trained model
model = joblib.load('app_rating_model.pkl')

# Define possible values for categorical encoding
update_months = {f"{i+1}": i+1 for i in range(12)}  # Assuming months are 1-12
types = {'free': 0, 'paid': 1}

@app.route('/')
def index():
    return render_template('form.html')  # Ensure your HTML file is named form.html

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve and convert form data
    size = float(request.form['size'])  # Convert size to float
    type_ = types.get(request.form['type'].lower(), 0)  # Convert type to numerical (0 for free, 1 for paid)
    update_month = update_months.get(request.form['update_month'], 1)  # Default to 1 if not found
    update_year = int(request.form['update_year'])  # Convert to integer
    price = float(request.form['price'])  # Log-transformed price
    reviews = float(request.form['reviews'])  # Log-transformed reviews
    installs = float(request.form['installs'])# Log-transformed installs

    # Prepare the one-hot encoded category and content rating features
    category_features = [int(request.form.get(f'Category_{i}', 0)) for i in range(1, 33)]  # Assuming 32 category columns
    content_rating_features = [int(request.form.get(f'Content Rating_{i}', 0)) for i in range(1, 6)]  # Assuming 5 content ratings

    price_log = np.log(price + 1)  # +1 to avoid log(0) if the price is 0
    reviews_log = np.log(reviews + 1)
    installs_log = np.log(installs + 1)

    # Combine all features into a single array
    features = np.array([[size, type_, update_month, update_year, price_log, reviews_log, installs_log] + category_features + content_rating_features])

    # Perform the prediction
    prediction = model.predict(features)  # Make sure the model can accept these features

    return f'Predicted Rating: {prediction[0]}'  # Return the first prediction result

if __name__ == '__main__':
    app.run(debug=True)

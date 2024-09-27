from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('app_rating_model.pkl')

types = {'free': 0, 'paid': 1}
update_months = {f"{i+1}": i+1 for i in range(12)}  # Assuming months 1-12

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    size = float(request.form['size'])
    type_ = types.get(request.form['type'].lower(), 0)
    update_month = update_months.get(request.form['update_month'], 1)
    update_year = int(request.form['update_year'])
    price = float(request.form['price'])
    reviews = float(request.form['reviews'])
    installs = float(request.form['installs'])

    price_log = np.log(price + 1)
    reviews_log = np.log(reviews + 1)
    installs_log = np.log(installs + 1)

    # Assume category and content rating are collected in similar ways
    category_features = [int(request.form.get(f'Category_{i}', 0)) for i in range(1, 33)]
    content_rating_features = [int(request.form.get(f'Content Rating_{i}', 0)) for i in range(1, 6)]

    features = np.array([[size, type_, update_month, update_year, price_log, reviews_log, installs_log] + category_features + content_rating_features])

    # Perform prediction
    prediction = model.predict(features)[0]

    # Render the result.html template with the prediction result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

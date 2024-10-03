from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('best_gb_model.pkl')
transformer = joblib.load('power_transformer.pkl')

types = {'free': 0, 'paid': 1}
update_months = {f"{i+1}": i+1 for i in range(12)}

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    size_MB = float(request.form['size'])
    size_kb = size_MB * 1024
    type_ = types.get(request.form['type'].lower(), 0)
    update_month = update_months.get(request.form['update_month'], 1)
    update_year = int(request.form['update_year'])
    price = float(request.form['price'])
    reviews = float(request.form['reviews'])
    installs = float(request.form['installs'])

    dummy_rating = 0

    features_to_transform = np.array([[reviews, installs, price, dummy_rating]])
    transformed_features_with_dummy = transformer.transform(features_to_transform)

    reviews_transformed = transformed_features_with_dummy[0, 0]
    installs_transformed = transformed_features_with_dummy[0, 1]
    price_transformed = transformed_features_with_dummy[0, 2]

    category_features = [int(request.form.get(f'Category_{i}', 0)) for i in range(1, 33)]
    content_rating_features = [int(request.form.get(f'Content Rating_{i}', 0)) for i in range(1, 6)]

    features = np.array([[size_kb, type_, update_month, update_year, price_transformed, reviews_transformed, installs_transformed] + category_features + content_rating_features])

    prediction_transformed = model.predict(features)[0]
    inverse_transform_input = np.array([[0, 0, 0, prediction_transformed]])
    predicted_rating_original = transformer.inverse_transform(inverse_transform_input)[0, 3]

    return render_template('result.html', prediction=predicted_rating_original)

if __name__ == '__main__':
    app.run(debug=True)

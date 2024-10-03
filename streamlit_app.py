import streamlit as st
import joblib
import numpy as np

model = joblib.load('best_gb_model.pkl')
transformer = joblib.load('power_transformer.pkl')

types = {'free': 0, 'paid': 1}
update_months = {f"{i+1}": i+1 for i in range(12)}

st.title("📊 App Rating Prediction")

st.markdown("""
<style>
    
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        padding: 15px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 10px 0;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
    .input-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .stNumberInput, .stSelectbox, .stTextInput {
        margin-bottom: 15px;
    }
    h2 {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Create the input form
with st.form(key='prediction_form'):
    st.header("Enter App Details")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            size_MB = st.number_input("Size in MB:", step=100)
            type_ = st.selectbox("Type:", options=["Free", "Paid"])
            update_month = st.number_input("Update Month:", min_value=1, max_value=12)
            reviews = st.number_input("Reviews:", step=100)
            price = st.number_input("Price:",min_value=0, step=1)

        with col2:
            update_year = st.number_input("Update Year:", min_value=2000, step=1)
            installs = st.number_input("Installs:", min_value=100 , step=100)
            category = st.selectbox("Category:", [
                "Art & Design", "Auto & Vehicles", "Beauty", "Books & Reference",
                "Business", "Comics", "Communication", "Dating", "Education",
                "Entertainment", "Events", "Finance", "Food & Drink",
                "Health & Fitness", "House & Home", "Libraries & Demo",
                "Lifestyle", "Game", "Family", "Medical", "Social", "Shopping",
                "Photography", "Sports", "Travel & Local", "Tools",
                "Personalization", "Productivity", "Parenting", "Weather",
                "Video Players", "News & Magazines", "Maps & Navigation"
            ])
            content_rating = st.selectbox("Content Rating:", [
                "Everyone", "Teen", "Everyone 10+", "Mature 17+", "Adults only 18+"
            ])
    
 
    submit_button = st.form_submit_button("Predict Rating")

if submit_button:
    size_kb = size_MB * 1024
    type_ = types.get(type_.lower(), 0)
    update_month = update_months.get(str(update_month), 1)
    price = float(price)
    reviews = float(reviews)
    installs = float(installs)

    dummy_rating = 0

    features_to_transform = np.array([[reviews, installs, price, dummy_rating]])
    transformed_features_with_dummy = transformer.transform(features_to_transform)

    reviews_transformed = transformed_features_with_dummy[0, 0]
    installs_transformed = transformed_features_with_dummy[0, 1]
    price_transformed = transformed_features_with_dummy[0, 2]

    category_features = [int(category == cat) for cat in [
        "Auto & Vehicles", "Beauty", "Books & Reference",
        "Business", "Comics", "Communication", "Dating", "Education",
        "Entertainment", "Events", "Finance", "Food & Drink",
        "Health & Fitness", "House & Home", "Libraries & Demo",
        "Lifestyle", "Game", "Family", "Medical", "Social", "Shopping",
        "Photography", "Sports", "Travel & Local", "Tools",
        "Personalization", "Productivity", "Parenting", "Weather",
        "Video Players", "News & Magazines", "Maps & Navigation"
    ]]

    content_rating_features = [int(content_rating == rating) for rating in [
        "Everyone", "Teen", "Everyone 10+", "Mature 17+", "Adults only 18+"
    ]]

    features = np.array([[size_kb, type_, update_month, update_year, price_transformed, reviews_transformed, installs_transformed] + category_features + content_rating_features])

    prediction_transformed = model.predict(features)[0]
    inverse_transform_input = np.array([[0, 0, 0, prediction_transformed]])
    predicted_rating_original = transformer.inverse_transform(inverse_transform_input)[0, 3]

 
    st.success(f"The predicted rating for the app is: **{predicted_rating_original:.2f}**")

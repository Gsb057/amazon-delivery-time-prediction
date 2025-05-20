import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
from src.utils import setup_logging
import os

# Setup logging
setup_logging()

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'best_model.pkl')
        if not os.path.exists(model_path):
            st.error("Model not found. Please train the model first.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(page_title="Delivery Time Predictor", layout="wide")
    
    st.title("Amazon Delivery Time Prediction")
    st.write("""
    This app predicts delivery times based on order details, agent information, 
    and environmental factors.
    """)
    
    model = load_model()
    if model is None:
        return
    
    with st.form("delivery_form"):
        st.header("Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            store_lat = st.number_input("Store Latitude", value=12.9716)
            store_long = st.number_input("Store Longitude", value=77.5946)
            agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
            agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)
        
        with col2:
            drop_lat = st.number_input("Drop Latitude", value=13.0827)
            drop_long = st.number_input("Drop Longitude", value=80.2707)
            weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Foggy"])
            traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
        
        vehicle = st.selectbox("Vehicle Type", ["Bike", "Truck", "Car"])
        area = st.selectbox("Area Type", ["Urban", "Metropolitan"])
        product_category = st.selectbox("Product Category", ["Electronics", "Grocery", "Fashion", "Furniture"])
        
        submitted = st.form_submit_button("Predict Delivery Time")
    
    if submitted:
        try:
            # Calculate distance
            distance = geodesic((store_lat, store_long), (drop_lat, drop_long)).km

            # Define all possible categories (from training time)
            weather_options = ["Sunny", "Rainy", "Cloudy", "Foggy"]
            traffic_options = ["Low", "Medium", "High", "Jam"]
            vehicle_options = ["Bike", "Truck", "Car"]
            area_options = ["Urban", "Metropolitan"]
            category_options = ["Electronics", "Grocery", "Fashion", "Furniture"]

            # Build input data
            input_data = {
                'Agent_Age': agent_age,
                'Agent_Rating': agent_rating,
                'Distance_km': distance
            }

            for option in weather_options:
                input_data[f"Weather_{option}"] = 1 if weather == option else 0
            for option in traffic_options:
                input_data[f"Traffic_{option}"] = 1 if traffic == option else 0
            for option in vehicle_options:
                input_data[f"Vehicle_{option}"] = 1 if vehicle == option else 0
            for option in area_options:
                input_data[f"Area_{option}"] = 1 if area == option else 0
            for option in category_options:
                input_data[f"Category_{option}"] = 1 if product_category == option else 0

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure all required columns are present
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns to match model input
            input_df = input_df[model.feature_names_in_]

            # Make prediction
            prediction = model.predict(input_df)[0]
            st.success(f"**Predicted Delivery Time:** {prediction:.2f} hours")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()

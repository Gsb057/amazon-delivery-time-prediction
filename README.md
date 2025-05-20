Amazon Delivery Time Prediction

This project was created as part of a GUVI class assignment based on a provided problem statement, with assistance from ChatGPT for code structuring, debugging, and guidance.

Overview

This project builds a machine learning pipeline to predict Amazon delivery times based on:

* Order details
* Agent demographics
* Location information
* Environmental factors (traffic, weather)

A full ML pipeline is included: preprocessing, feature engineering, model training, evaluation, and a live Streamlit app for delivery time prediction.

Project Structure

amazon_delivery_prediction/
├── src/                      # Core Python scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
│
├── app.py                   # Streamlit app
├── README.md                # Project overview (this file)
├── requirements.txt         # Python dependencies

How to Run This Project

1. Clone or Download the Repo

2. Install Requirements

 -pip install -r requirements.txt

3. Run the Full ML Pipeline (Optional)

 -python src/data_preprocessing.py
 -python src/feature_engineering.py
 -python src/model_training.py

 Skip these steps if you're only using the trained model (best_model.pkl).
 To download the trainded model : https://drive.google.com/drive/folders/1GWELlTSPZ_YC_k2kPgWXu-4EqqkFZZd1?usp=drive_link

4. Launch the Streamlit App

 -streamlit run app.py

Features Implemented

* Data cleaning and preprocessing
* Feature engineering:

  * Geodesic distance
  * Time delta calculation
  * One-hot encoding
* Multiple ML models trained:

  * Linear Regression
  * Random Forest
  * Gradient Boosting
  * XGBoost (best performer)
* Model evaluation (RMSE, MAE, R2)
* Feature importance chart
* Streamlit-based prediction UI

Credits

This project was completed as part of a GUVI class assignment. Thanks to:

* ChatGPT for real-time coding and debugging guidance
* GUVI platform for the problem statement and learning path

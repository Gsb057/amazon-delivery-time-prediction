import logging
import sys
from utils import setup_logging, log_step, save_data
from geopy.distance import geodesic
import pandas as pd
from datetime import datetime

def calculate_distance(df):
    log_step("calculating distances")
    df['Distance_km'] = df.apply(
        lambda row: geodesic(
            (row['Store_Latitude'], row['Store_Longitude']),
            (row['Drop_Latitude'], row['Drop_Longitude'])
        ).km,
        axis=1
    )
    logging.info(f"Distance range: {df['Distance_km'].min():.2f} to {df['Distance_km'].max():.2f} km")
    return df

def extract_time_features(df):
    log_step("extracting time features")

    # Drop rows with missing date or time before conversion
    df = df.dropna(subset=['Order_Date', 'Order_Time', 'Pickup_Time'])

    # Convert to datetime (errors='coerce' handles any unexpected formats)
    df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce')
    df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Pickup_Time'], errors='coerce')

    # Drop rows where datetime conversion failed
    df = df.dropna(subset=['Order_DateTime', 'Pickup_DateTime'])

    # Extract time features
    df['Order_Hour'] = df['Order_DateTime'].dt.hour
    df['Order_Day'] = df['Order_DateTime'].dt.dayofweek
    df['Pickup_Hour'] = df['Pickup_DateTime'].dt.hour
    df['Order_to_Pickup_Min'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60

    logging.info("Added time-based features")
    return df

def encode_categorical(df):
    log_step("encoding categorical features")

    # Define fixed categories (important!)
    categories = {
        'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Foggy'],
        'Traffic': ['Low', 'Medium', 'High', 'Jam'],
        'Vehicle': ['Bike', 'Truck', 'Car'],
        'Area': ['Urban', 'Metropolitan'],
        'Category': ['Electronics', 'Grocery', 'Fashion', 'Furniture']
    }

    for col, values in categories.items():
        for val in values:
            col_name = f"{col}_{val}"
            df[col_name] = (df[col] == val).astype(int)

    df.drop(columns=categories.keys(), inplace=True)
    logging.info(f"Encoded {len(categories)} categorical variables with fixed one-hot encoding")
    return df

if __name__ == "__main__":
    setup_logging()
    try:
        df = pd.read_csv('data/processed_data.csv')
        df = calculate_distance(df)
        df = extract_time_features(df)
        df = encode_categorical(df)
        save_data(df, 'data/final_data.csv')
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}", exc_info=True)
        sys.exit(1)

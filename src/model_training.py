from utils import setup_logging, log_step
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
import os
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data():
    log_step("preparing data for modeling")
    df = pd.read_csv('data/final_data.csv')

    # Drop unused or ID/time columns
    X = df.drop(['Delivery_Time', 'Order_ID', 'Order_Date', 'Order_Time',
             'Pickup_Time', 'Order_DateTime', 'Pickup_DateTime'], axis=1, errors='ignore')
    y = df['Delivery_Time']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    log_step(f"training {name}")

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name.lower().replace(" ", "_"))

        logging.info(f"{name} performance:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        return model, metrics

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance.values, y=importance.index)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()

        os.makedirs('models', exist_ok=True)
        plt.savefig('models/feature_importance.png')
        plt.close()
        logging.info("Saved feature importance plot to models/feature_importance.png")

if __name__ == "__main__":
    setup_logging()
    mlflow.set_experiment("Amazon_Delivery_Time_Prediction")

    try:
        X_train, X_test, y_train, y_test = prepare_data()

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        }

        best_model = None
        best_r2 = -float('inf')

        for name, model in models.items():
            trained_model, metrics = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model = trained_model
                best_model_name = name

        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.pkl')
        logging.info(f"Saved best model ({best_model_name}, R2: {best_r2:.4f}) to models/best_model.pkl")

        # Plot feature importance
        plot_feature_importance(best_model, X_train.columns)

    except Exception as e:
        logging.error(f"Error in model training: {str(e)}", exc_info=True)
        sys.exit(1)

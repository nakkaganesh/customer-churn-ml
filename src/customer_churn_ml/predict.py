from pathlib import Path

import joblib
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Saved model path
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Load trained pipeline
model = joblib.load(MODEL_PATH)

# Candidate business threshold
THRESHOLD = 0.30


def predict_churn(customer_data: dict) -> dict:
    """Predict churn probability for one customer."""

    customer = pd.DataFrame([customer_data])

    churn_probability = model.predict_proba(customer)[0, 1]

    prediction = (
        "Yes"
        if churn_probability >= THRESHOLD
        else "No"
    )

    return {
        "churn_probability": float(churn_probability),
        "prediction": prediction,
    }


if __name__ == "__main__":

    new_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 475.0,
    }

    result = predict_churn(new_customer)

    print(
        f"Churn Probability: "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Prediction: {result['prediction']}"
    )
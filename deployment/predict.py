# ============================================================
# predict.py
#
# This file loads the trained machine learning pipeline
# and provides a function for making predictions.
# ============================================================

import joblib
import pandas as pd

# ------------------------------------------------------------
# Load the trained machine learning pipeline
# ------------------------------------------------------------

MODEL_PATH = "models/heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------

def predict_heart_disease(input_data):
    """
    Predict heart disease using the trained pipeline.
    """

    # Convert dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict class
    prediction = int(model.predict(input_df)[0])

    # Predict confidence
    confidence = float(model.predict_proba(input_df)[0].max())

    return prediction, round(confidence, 4)
# ============================================================
# app.py
#
# Main FastAPI application
# ============================================================

from datetime import UTC, datetime

from fastapi import FastAPI

from deployment.predict import predict_heart_disease
from deployment.schemas import HeartDiseaseInput, HeartDiseaseResponse

# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease using a trained Random Forest pipeline",
    version="1.0",
)

# ------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------


@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API is running successfully.",
    }


# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------


@app.post("/predict", response_model=HeartDiseaseResponse)
def predict(data: HeartDiseaseInput):
    """
    Predict whether the patient has heart disease.
    """

    prediction, confidence = predict_heart_disease(data.model_dump())

    prediction_label = (
        "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "model_version": "1.0",
        "timestamp": datetime.now(UTC).isoformat(),
    }
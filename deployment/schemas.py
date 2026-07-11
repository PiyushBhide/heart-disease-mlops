# ============================================================
# schemas.py
#
# This file defines the structure of the input and output data
# used by the FastAPI application.
# ============================================================

from pydantic import BaseModel


class HeartDiseaseInput(BaseModel):
    """
    Input schema for heart disease prediction.
    """

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


class HeartDiseaseResponse(BaseModel):
    """
    Response schema returned after a successful prediction.
    """

    prediction: int
    prediction_label: str
    confidence: float
    model_version: str
    timestamp: str
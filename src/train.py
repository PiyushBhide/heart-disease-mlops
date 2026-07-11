from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_processing import (
    create_train_test_split,
    load_data,
    split_features_target,
)


NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def build_model_pipeline() -> Pipeline:
    """
    Build the complete preprocessing and Random Forest pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline containing preprocessing and classification steps.
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    return model_pipeline


def train_model(
    dataset_path: str | Path,
) -> tuple[Pipeline, dict[str, float]]:
    """
    Load the dataset, split it, train the model, and calculate metrics.

    Parameters
    ----------
    dataset_path:
        Path to the cleaned heart-disease dataset.

    Returns
    -------
    tuple
        Trained model pipeline and evaluation metrics.
    """

    data = load_data(dataset_path)

    features, target = split_features_target(data)

    X_train, X_test, y_train, y_test = create_train_test_split(
        features=features,
        target=target,
        test_size=0.20,
        random_state=42,
    )

    model_pipeline = build_model_pipeline()

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    prediction_probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, prediction_probabilities)
        ),
    }

    return model_pipeline, metrics


def save_model(
    model_pipeline: Pipeline,
    output_path: str | Path,
) -> Path:
    """
    Save a trained model pipeline using Joblib.

    Parameters
    ----------
    model_pipeline:
        Trained Scikit-learn pipeline.

    output_path:
        Path where the model should be saved.

    Returns
    -------
    pathlib.Path
        Final saved model path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_pipeline,
        output_path,
    )

    return output_path


def run_training() -> None:
    """
    Run complete model training from the project root.
    """

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "heart_cleaned.csv"
    )

    model_output_path = (
        project_root
        / "models"
        / "heart_disease_pipeline.joblib"
    )

    model_pipeline, metrics = train_model(dataset_path)

    saved_path = save_model(
        model_pipeline,
        model_output_path,
    )

    print("Model training completed successfully.")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Model saved to: {saved_path}")


if __name__ == "__main__":
    run_training()
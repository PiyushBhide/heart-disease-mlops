from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_processing import (
    load_data,
    split_features_target,
)
from src.train import (
    build_model_pipeline,
    save_model,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_cleaned.csv"
)


def test_build_model_pipeline_returns_pipeline():
    """
    Verify that the model builder returns a Scikit-learn Pipeline.
    """

    pipeline = build_model_pipeline()

    assert isinstance(pipeline, Pipeline)


def test_pipeline_contains_required_steps():
    """
    Verify that preprocessing and classifier steps exist.
    """

    pipeline = build_model_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_classifier_is_random_forest():
    """
    Verify that the selected classifier is Random Forest.
    """

    pipeline = build_model_pipeline()

    classifier = pipeline.named_steps["classifier"]

    assert isinstance(
        classifier,
        RandomForestClassifier,
    )


def test_model_can_be_trained():
    """
    Verify that the pipeline can fit the cleaned dataset.
    """

    data = load_data(DATASET_PATH)

    features, target = split_features_target(data)

    pipeline = build_model_pipeline()

    pipeline.fit(features, target)

    assert hasattr(
        pipeline.named_steps["classifier"],
        "classes_",
    )


def test_model_can_generate_predictions():
    """
    Verify that the trained model produces predictions.
    """

    data = load_data(DATASET_PATH)

    features, target = split_features_target(data)

    pipeline = build_model_pipeline()

    pipeline.fit(features, target)

    predictions = pipeline.predict(
        features.head(5)
    )

    assert len(predictions) == 5

    assert set(predictions).issubset(
        {0, 1}
    )


def test_model_can_generate_probabilities():
    """
    Verify that prediction probabilities are valid.
    """

    data = load_data(DATASET_PATH)

    features, target = split_features_target(data)

    pipeline = build_model_pipeline()

    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(
        features.head(5)
    )

    assert probabilities.shape == (5, 2)

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_training_metrics_are_valid():
    """
    Verify that training returns valid evaluation metrics.
    """

    _, metrics = train_model(DATASET_PATH)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_model_accuracy_is_acceptable():
    """
    Verify that model accuracy is above a basic threshold.
    """

    _, metrics = train_model(DATASET_PATH)

    assert metrics["accuracy"] >= 0.80


def test_model_roc_auc_is_acceptable():
    """
    Verify that model ROC-AUC is above a basic threshold.
    """

    _, metrics = train_model(DATASET_PATH)

    assert metrics["roc_auc"] >= 0.85


def test_save_model_creates_file(tmp_path):
    """
    Verify that the trained model can be saved.
    """

    pipeline, _ = train_model(DATASET_PATH)

    output_path = (
        tmp_path
        / "test_model.joblib"
    )

    saved_path = save_model(
        pipeline,
        output_path,
    )

    assert saved_path.exists()


def test_saved_model_can_be_loaded(tmp_path):
    """
    Verify that a saved model can be loaded and used.
    """

    pipeline, _ = train_model(DATASET_PATH)

    output_path = (
        tmp_path
        / "test_model.joblib"
    )

    save_model(
        pipeline,
        output_path,
    )

    loaded_model = joblib.load(output_path)

    data = load_data(DATASET_PATH)

    features, _ = split_features_target(data)

    predictions = loaded_model.predict(
        features.head(3)
    )

    assert len(predictions) == 3
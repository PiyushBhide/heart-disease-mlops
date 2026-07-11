from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import (
    EXPECTED_COLUMNS,
    create_train_test_split,
    load_data,
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "heart_cleaned.csv"


def test_dataset_file_exists():
    """
    Verify that the cleaned dataset exists in the expected location.
    """
    assert DATASET_PATH.exists(), (
        f"Cleaned dataset was not found at {DATASET_PATH}"
    )


def test_load_data_returns_dataframe():
    """
    Verify that load_data returns a non-empty Pandas DataFrame.
    """
    data = load_data(DATASET_PATH)

    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_dataset_contains_expected_columns():
    """
    Verify that every required project column exists.
    """
    data = load_data(DATASET_PATH)

    for column in EXPECTED_COLUMNS:
        assert column in data.columns


def test_cleaned_dataset_has_no_missing_values():
    """
    Verify that the cleaned dataset contains no missing values.
    """
    data = load_data(DATASET_PATH)

    total_missing_values = int(data.isnull().sum().sum())

    assert total_missing_values == 0


def test_split_features_target():
    """
    Verify that the target is separated correctly from the features.
    """
    data = load_data(DATASET_PATH)

    features, target = split_features_target(data)

    assert "target" not in features.columns
    assert target.name == "target"
    assert len(features) == len(target)
    assert features.shape[1] == 13


def test_train_test_split_sizes():
    """
    Verify the expected 80/20 train-test split sizes.
    """
    data = load_data(DATASET_PATH)
    features, target = split_features_target(data)

    X_train, X_test, y_train, y_test = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    assert X_train.shape == (242, 13)
    assert X_test.shape == (61, 13)
    assert y_train.shape == (242,)
    assert y_test.shape == (61,)


def test_train_test_split_is_reproducible():
    """
    Verify that the same random state creates the same split.
    """
    data = load_data(DATASET_PATH)
    features, target = split_features_target(data)

    first_split = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    second_split = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    for first_object, second_object in zip(
        first_split,
        second_split,
        strict=True,
    ):
        assert first_object.equals(second_object)


def test_load_data_raises_error_for_missing_file(tmp_path):
    """
    Verify that a clear error is raised for a missing CSV file.
    """
    missing_file = tmp_path / "missing_dataset.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_split_features_target_raises_error_without_target():
    """
    Verify that an error is raised when the target column is absent.
    """
    sample_data = pd.DataFrame(
        {
            "age": [45, 60],
            "chol": [220, 250],
        }
    )

    with pytest.raises(ValueError, match="Target column"):
        split_features_target(sample_data)


def test_invalid_test_size_raises_error():
    """
    Verify that test_size must be between zero and one.
    """
    data = load_data(DATASET_PATH)
    features, target = split_features_target(data)

    with pytest.raises(ValueError, match="test_size"):
        create_train_test_split(
            features,
            target,
            test_size=1.5,
        )
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target"

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the cleaned heart-disease dataset from a CSV file.

    Parameters
    ----------
    file_path:
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded and validated dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If expected columns are missing or the dataset is empty.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the input features from the target column.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    features = data.drop(columns=[target_column])
    target = data[target_column]

    return features, target


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create reproducible training and testing datasets.
    """
    if len(features) != len(target):
        raise ValueError(
            "Features and target must contain the same number of rows."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
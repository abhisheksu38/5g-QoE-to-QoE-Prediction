"""
Data preprocessing utilities for the 5G QoS and QoE prediction project.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load the processed 5G QoS/QoE dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


def clean_dataset(df):
    """
    Basic cleaning of the dataset.

    - Removes duplicate rows.
    - Converts infinite values to NaN.
    - Keeps the original dataframe structure.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def get_qoe_column(df):
    """
    Identify the QoE target column.

    The current project dataset uses 'qoe'.

    Returns
    -------
    str
        QoE column name.
    """

    if "qoe" in df.columns:
        return "qoe"

    raise KeyError(
        "QoE column 'qoe' was not found in the dataset."
    )


def prepare_qoe_dataset(df):
    """
    Prepare the dataframe for QoE modelling.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe containing a valid QoE target.
    """

    df = clean_dataset(df)

    qoe_column = get_qoe_column(df)

    # Remove rows where the target is missing
    df = df.dropna(subset=[qoe_column])

    return df.reset_index(drop=True)


def get_basic_statistics(df):
    """
    Return basic dataset statistics.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
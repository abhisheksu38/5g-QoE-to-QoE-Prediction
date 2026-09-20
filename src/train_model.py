"""
Machine learning training pipeline for 5G QoE prediction.

Models:
- Random Forest
- Gradient Boosting

The Gradient Boosting model is used as the final model because
it performed better during the project validation.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .feature_engineering import (
    MODEL_FEATURES,
    validate_features
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "5g_qos_qoe_features.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():
    """
    Load the processed QoS/QoE dataset.
    """

    df = pd.read_csv(DATA_PATH)

    validate_features(df)

    if "qoe" not in df.columns:
        raise KeyError(
            "Target column 'qoe' was not found."
        )

    df = df.dropna(
        subset=MODEL_FEATURES + ["qoe"]
    )

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_training_data(df):
    """
    Separate input features and QoE target.
    """

    X = df[MODEL_FEATURES].copy()

    y = df["qoe"].copy()

    return X, y


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(X_train, y_train):
    """
    Train Random Forest and Gradient Boosting models.
    """

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    random_forest.fit(
        X_train,
        y_train
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )

    return random_forest, gradient_boosting


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate MAE, RMSE and R².
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_pipeline():
    """
    Complete model training pipeline.

    Returns
    -------
    dict
        Trained models and evaluation metrics.
    """

    print("=" * 60)
    print("5G QoE MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_training_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y = prepare_training_data(df)

    print(
        f"Number of features: {len(MODEL_FEATURES)}"
    )

    print(
        f"Number of samples: {len(X)}"
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    random_forest, gradient_boosting = train_models(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate Random Forest
    # --------------------------------------------------------

    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    print("\nRandom Forest")
    print("-" * 30)

    for name, value in rf_metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Evaluate Gradient Boosting
    # --------------------------------------------------------

    gb_metrics = evaluate_model(
        gradient_boosting,
        X_test,
        y_test
    )

    print("\nGradient Boosting")
    print("-" * 30)

    for name, value in gb_metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Save Gradient Boosting model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "qoe_gradient_boosting.pkl"
    )

    joblib.dump(
        gradient_boosting,
        model_path
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_path = (
        MODEL_DIR
        / "qoe_features.pkl"
    )

    joblib.dump(
        MODEL_FEATURES,
        feature_path
    )

    print("\nModels saved:")
    print(model_path)
    print(feature_path)

    print("\nTraining completed successfully.")

    return {
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
        "random_forest_metrics": rf_metrics,
        "gradient_boosting_metrics": gb_metrics
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    train_pipeline()
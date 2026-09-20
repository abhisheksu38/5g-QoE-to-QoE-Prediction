"""
Time-Series QoE Forecasting
---------------------------

5G QoE forecasting using a lag-based Gradient Boosting model.

Dashboard architecture:

5G QoS Inputs
      |
      v
QoE Prediction
      |
      v
Current Predicted QoE
      |
      v
Conditioned QoE Forecast

Forecast Validation:
Historical QoE
      |
      v
Source-file-aware chronological validation
      |
      v
Actual vs Predicted QoE
      |
      v
MAE / RMSE / R²
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ================================================================
# PREPARE QoE SERIES
# ================================================================

def prepare_qoe_series(qoe_series):
    """
    Prepare a complete QoE series for forecasting.

    The overall forecasting pipeline requires at least
    20 observations.
    """

    series = pd.to_numeric(
        qoe_series,
        errors="coerce"
    ).dropna().reset_index(drop=True)

    if len(series) < 20:

        raise ValueError(
            "At least 20 QoE observations are required "
            "for forecasting."
        )

    return series


# ================================================================
# PREPARE QoE SERIES FOR LAG FEATURES
# ================================================================

def prepare_series_for_lag_features(qoe_series):
    """
    Prepare a QoE series specifically for lag-feature creation.

    This function intentionally does NOT require 20 observations.

    A source file only needs enough observations to create:
        - lag features
        - rolling mean 3
        - rolling mean 5
        - rolling standard deviation 5

    The caller decides the minimum required length.
    """

    series = pd.to_numeric(
        qoe_series,
        errors="coerce"
    ).dropna().reset_index(drop=True)

    return series


# ================================================================
# FORECAST FEATURE NAMES
# ================================================================

def get_forecast_features(
    n_lags=5
):

    feature_columns = [
        f"lag_{i}"
        for i in range(
            1,
            n_lags + 1
        )
    ]

    feature_columns += [
        "rolling_mean_3",
        "rolling_mean_5",
        "rolling_std_5"
    ]

    return feature_columns


# ================================================================
# CREATE LAG FEATURES
# ================================================================

def create_lag_features(
    series,
    n_lags=5
):
    """
    Create lag and rolling features.

    IMPORTANT:
    This function does not require 20 observations.
    """

    series = prepare_series_for_lag_features(
        series
    )

    if len(series) < (
        n_lags + 2
    ):

        raise ValueError(
            f"At least {n_lags + 2} QoE observations "
            "are required to create lag features."
        )

    df = pd.DataFrame(
        {
            "qoe": series
        }
    )

    # ------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------

    for lag in range(
        1,
        n_lags + 1
    ):

        df[
            f"lag_{lag}"
        ] = (
            df["qoe"]
            .shift(lag)
        )

    # ------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------

    df["rolling_mean_3"] = (
        df["qoe"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["rolling_mean_5"] = (
        df["qoe"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    df["rolling_std_5"] = (
        df["qoe"]
        .shift(1)
        .rolling(5)
        .std()
    )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    return df


# ================================================================
# TRAIN FORECASTING MODEL
# ================================================================

def train_forecasting_model(
    series,
    n_lags=5
):

    series = prepare_qoe_series(
        series
    )

    df = create_lag_features(
        series,
        n_lags=n_lags
    )

    feature_columns = get_forecast_features(
        n_lags=n_lags
    )

    X = df[
        feature_columns
    ]

    y = df["qoe"]

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    split_index = int(
        len(df) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.clip(
        predictions,
        0,
        100
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return (
        model,
        feature_columns,
        metrics,
        df
    )


# ================================================================
# CREATE FORECAST INPUT
# ================================================================

def create_forecast_input(
    history,
    n_lags=5
):

    history = list(history)

    if len(history) < n_lags:

        raise ValueError(
            f"At least {n_lags} historical values "
            "are required."
        )

    recent = np.asarray(
        history,
        dtype=float
    )

    row = {}

    # ------------------------------------------------------------
    # Lag values
    # ------------------------------------------------------------

    for lag in range(
        1,
        n_lags + 1
    ):

        row[
            f"lag_{lag}"
        ] = recent[-lag]

    # ------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------

    row["rolling_mean_3"] = np.mean(
        recent[-3:]
    )

    row["rolling_mean_5"] = np.mean(
        recent[-5:]
    )

    row["rolling_std_5"] = np.std(
        recent[-5:]
    )

    return pd.DataFrame(
        [row]
    )


# ================================================================
# RAW RECURSIVE FORECAST
# ================================================================

def _raw_recursive_forecast(
    model,
    feature_columns,
    history,
    horizon=10,
    n_lags=5
):

    history = list(
        history
    )

    raw_forecast = []

    for _ in range(
        horizon
    ):

        input_df = create_forecast_input(
            history,
            n_lags=n_lags
        )

        input_df = input_df[
            feature_columns
        ]

        prediction = float(
            model.predict(
                input_df
            )[0]
        )

        prediction = float(
            np.clip(
                prediction,
                0,
                100
            )
        )

        raw_forecast.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.asarray(
        raw_forecast
    )


# ================================================================
# ANCHOR FORECAST TO CURRENT QoE
# ================================================================

def anchor_forecast(
    current_qoe,
    raw_forecast,
    anchor_strength=0.70
):
    """
    Smooth the transition from the current QoE
    to the learned future forecast.

    A stronger anchor keeps the first future
    prediction closer to the current QoE.
    """

    raw_forecast = np.asarray(
        raw_forecast,
        dtype=float
    )

    if len(raw_forecast) == 0:

        return raw_forecast

    current_qoe = float(
        np.clip(
            current_qoe,
            0,
            100
        )
    )

    anchored = []

    horizon = len(
        raw_forecast
    )

    for i, raw_value in enumerate(
        raw_forecast
    ):

        if horizon == 1:

            decay = (
                1
                - anchor_strength
            )

        else:

            decay = (
                (1 - anchor_strength)
                +
                (
                    anchor_strength
                    *
                    (
                        i
                        /
                        (
                            horizon - 1
                        )
                    )
                )
            )

        value = (
            current_qoe
            * (1 - decay)
            +
            raw_value
            * decay
        )

        anchored.append(
            float(
                np.clip(
                    value,
                    0,
                    100
                )
            )
        )

    # Explicitly ensure the first future value
    # stays close to the current predicted QoE.

    anchored[0] = float(
        np.clip(
            (
                current_qoe
                * anchor_strength
                +
                raw_forecast[0]
                * (1 - anchor_strength)
            ),
            0,
            100
        )
    )

    return np.asarray(
        anchored
    )


# ================================================================
# STANDARD FORECAST
# ================================================================

def forecast_qoe(
    qoe_series,
    horizon=10,
    n_lags=5
):

    series = prepare_qoe_series(
        qoe_series
    )

    (
        model,
        feature_columns,
        metrics,
        _
    ) = train_forecasting_model(
        series,
        n_lags=n_lags
    )

    history = list(
        series.astype(float)
    )

    raw_forecast = _raw_recursive_forecast(
        model=model,
        feature_columns=feature_columns,
        history=history,
        horizon=horizon,
        n_lags=n_lags
    )

    return (
        raw_forecast,
        metrics
    )


# ================================================================
# CONDITIONED FORECAST
# ================================================================

def forecast_qoe_from_current(
    qoe_series,
    current_qoe,
    horizon=10,
    n_lags=5
):
    """
    Generate a future QoE forecast using the
    current QoE predicted from the dashboard inputs.
    """

    series = prepare_qoe_series(
        qoe_series
    )

    (
        model,
        feature_columns,
        metrics,
        _
    ) = train_forecasting_model(
        series,
        n_lags=n_lags
    )

    current_qoe = float(
        np.clip(
            current_qoe,
            0,
            100
        )
    )

    # ------------------------------------------------------------
    # Historical context
    # ------------------------------------------------------------

    history = list(
        series
        .tail(20)
        .astype(float)
    )

    # Current dashboard prediction becomes
    # the newest QoE state.

    history.append(
        current_qoe
    )

    # ------------------------------------------------------------
    # Raw model forecast
    # ------------------------------------------------------------

    raw_forecast = _raw_recursive_forecast(
        model=model,
        feature_columns=feature_columns,
        history=history,
        horizon=horizon,
        n_lags=n_lags
    )

    # ------------------------------------------------------------
    # Anchor forecast to current predicted QoE
    # ------------------------------------------------------------

    forecast_values = anchor_forecast(
        current_qoe=current_qoe,
        raw_forecast=raw_forecast,
        anchor_strength=0.70
    )

    return (
        forecast_values,
        metrics
    )


# ================================================================
# TREND
# ================================================================

def calculate_trend(
    current_qoe,
    future_qoe
):

    change = (
        future_qoe
        - current_qoe
    )

    if change > 1:

        return "Improving"

    elif change < -1:

        return "Declining"

    else:

        return "Stable"


# ================================================================
# STANDARD FORECAST SUMMARY
# ================================================================

def forecast_summary(
    qoe_series,
    horizon=10,
    n_lags=5,
    window=20
):

    series = prepare_qoe_series(
        qoe_series
    )

    (
        forecast_values,
        metrics
    ) = forecast_qoe(
        series,
        horizon=horizon,
        n_lags=n_lags
    )

    current_qoe = float(
        series.iloc[-1]
    )

    future_qoe = float(
        forecast_values[-1]
    )

    change = (
        future_qoe
        - current_qoe
    )

    trend = calculate_trend(
        current_qoe,
        future_qoe
    )

    return {
        "current_qoe": current_qoe,
        "future_qoe": future_qoe,
        "change": change,
        "trend": trend,
        "forecast_values": forecast_values,
        "metrics": metrics
    }


# ================================================================
# CONDITIONED FORECAST SUMMARY
# ================================================================

def forecast_summary_from_current(
    qoe_series,
    current_qoe,
    horizon=10,
    n_lags=5
):

    current_qoe = float(
        np.clip(
            current_qoe,
            0,
            100
        )
    )

    (
        forecast_values,
        metrics
    ) = forecast_qoe_from_current(
        qoe_series=qoe_series,
        current_qoe=current_qoe,
        horizon=horizon,
        n_lags=n_lags
    )

    future_qoe = float(
        forecast_values[-1]
    )

    change = (
        future_qoe
        - current_qoe
    )

    trend = calculate_trend(
        current_qoe,
        future_qoe
    )

    return {
        "current_qoe": current_qoe,
        "future_qoe": future_qoe,
        "change": change,
        "trend": trend,
        "forecast_values": forecast_values,
        "metrics": metrics
    }


# ================================================================
# SOURCE-FILE-AWARE VALIDATION
# ================================================================

def get_grouped_validation_predictions(
    df,
    n_lags=5,
    train_ratio=0.8
):
    """
    Validate the forecasting model separately within
    each source file.

    This prevents QoE observations from different
    measurement sessions from being mixed together.
    """

    required_columns = [
        "qoe",
        "timestamp",
        "source_file"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Missing required columns: {missing}"
        )

    data = df.copy()

    # ------------------------------------------------------------
    # Convert timestamp
    # ------------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
        format="mixed"
    )

    # ------------------------------------------------------------
    # Convert QoE
    # ------------------------------------------------------------

    data["qoe"] = pd.to_numeric(
        data["qoe"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "qoe",
            "timestamp",
            "source_file"
        ]
    )

    # ------------------------------------------------------------
    # Chronological order inside each measurement file
    # ------------------------------------------------------------

    data = data.sort_values(
        [
            "source_file",
            "timestamp"
        ]
    ).reset_index(
        drop=True
    )

    all_actual = []
    all_predicted = []

    # ============================================================
    # VALIDATE EACH SOURCE FILE
    # ============================================================

    for source_file, group in data.groupby(
        "source_file",
        sort=False
    ):

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # We need enough observations to create:
        #
        # lag_1 ... lag_5
        # rolling_mean_3
        # rolling_mean_5
        # rolling_std_5
        #
        # Minimum practical requirement = 7 observations.
        # --------------------------------------------------------

        if len(group) < (
            n_lags + 2
        ):

            continue

        series = group[
            "qoe"
        ].reset_index(
            drop=True
        )

        # --------------------------------------------------------
        # Create lag features.
        #
        # IMPORTANT:
        # create_lag_features no longer requires 20 observations.
        # --------------------------------------------------------

        lag_df = create_lag_features(
            series,
            n_lags=n_lags
        )

        if len(lag_df) < 5:

            continue

        feature_columns = get_forecast_features(
            n_lags=n_lags
        )

        # --------------------------------------------------------
        # Chronological train/test split
        # --------------------------------------------------------

        split_index = int(
            len(lag_df)
            * train_ratio
        )

        if split_index <= 0:

            continue

        if split_index >= len(lag_df):

            continue

        X_train = (
            lag_df[
                feature_columns
            ]
            .iloc[:split_index]
        )

        y_train = (
            lag_df[
                "qoe"
            ]
            .iloc[:split_index]
        )

        X_test = (
            lag_df[
                feature_columns
            ]
            .iloc[split_index:]
        )

        y_test = (
            lag_df[
                "qoe"
            ]
            .iloc[split_index:]
        )

        if len(X_train) == 0:

            continue

        if len(X_test) == 0:

            continue

        # --------------------------------------------------------
        # Train forecasting model
        # --------------------------------------------------------

        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        # --------------------------------------------------------
        # Raw predictions
        # --------------------------------------------------------

        raw_predictions = model.predict(
            X_test
        )

        raw_predictions = np.clip(
            raw_predictions,
            0,
            100
        )

        # --------------------------------------------------------
        # Anchor validation predictions to the last training QoE.
        #
        # This is the same logic used in the dashboard forecast.
        # --------------------------------------------------------

        if len(y_train) > 0:

            anchor_qoe = float(
                y_train.iloc[-1]
            )

            predictions = anchor_forecast(
                current_qoe=anchor_qoe,
                raw_forecast=raw_predictions,
                anchor_strength=0.70
            )

        else:

            predictions = raw_predictions

        # --------------------------------------------------------
        # Store results
        # --------------------------------------------------------

        all_actual.extend(
            y_test.values
        )

        all_predicted.extend(
            predictions
        )

    # ============================================================
    # CHECK RESULTS
    # ============================================================

    if len(all_actual) == 0:

        raise ValueError(
            "No valid validation samples were generated."
        )

    actual = np.asarray(
        all_actual
    )

    predicted = np.asarray(
        all_predicted
    )

    # ============================================================
    # METRICS
    # ============================================================

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    # ============================================================
    # RESULT DATAFRAME
    # ============================================================

    result = pd.DataFrame(
        {
            "Actual QoE": actual,
            "Predicted QoE": predicted
        }
    )

    return (
        result,
        metrics
    )


# ================================================================
# STANDARD VALIDATION
# ================================================================

def get_validation_predictions(
    qoe_series,
    n_lags=5
):

    series = prepare_qoe_series(
        qoe_series
    )

    (
        model,
        feature_columns,
        metrics,
        df
    ) = train_forecasting_model(
        series,
        n_lags=n_lags
    )

    split_index = int(
        len(df) * 0.8
    )

    validation_df = df.iloc[
        split_index:
    ].copy()

    X_validation = (
        validation_df[
            feature_columns
        ]
    )

    actual = (
        validation_df[
            "qoe"
        ].values
    )

    raw_predictions = model.predict(
        X_validation
    )

    raw_predictions = np.clip(
        raw_predictions,
        0,
        100
    )

    anchor_qoe = float(
        df[
            "qoe"
        ].iloc[
            split_index - 1
        ]
    )

    predicted = anchor_forecast(
        current_qoe=anchor_qoe,
        raw_forecast=raw_predictions,
        anchor_strength=0.70
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    result = pd.DataFrame(
        {
            "Actual QoE": actual,
            "Predicted QoE": predicted
        }
    )

    return (
        result,
        metrics
    )


# ================================================================
# EVALUATE FORECASTING MODEL
# ================================================================

def evaluate_forecasting_model(
    qoe_series,
    n_lags=5
):

    (
        _,
        metrics
    ) = get_validation_predictions(
        qoe_series,
        n_lags=n_lags
    )

    return metrics
"""
Evaluate 5G QoE Forecasting Model
---------------------------------

Generates:

1. Forecast validation metrics
2. Actual vs Predicted QoE plot
3. Validation prediction CSV

Uses the same forecasting logic as the Streamlit dashboard.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from forecasting import (
    get_grouped_validation_predictions
)


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "5g_qos_qoe_features.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
)

METRICS_PATH = (
    RESULTS_PATH
    / "metrics"
)

PLOTS_PATH = (
    RESULTS_PATH
    / "plots"
)


# ================================================================
# CREATE DIRECTORIES
# ================================================================

METRICS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# LOAD DATASET
# ================================================================

print()
print("=" * 60)
print("5G QoE FORECASTING EVALUATION")
print("=" * 60)

dataset = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {dataset.shape}"
)


# ================================================================
# RUN VALIDATION
# ================================================================

validation_df, metrics = (
    get_grouped_validation_predictions(
        dataset,
        n_lags=5,
        train_ratio=0.8
    )
)


# ================================================================
# PRINT RESULTS
# ================================================================

print()
print(
    "FINAL QoE FORECASTING VALIDATION RESULTS"
)

print(
    f"Validation samples : "
    f"{len(validation_df)}"
)

print(
    f"MAE                : "
    f"{metrics['MAE']:.4f}"
)

print(
    f"RMSE               : "
    f"{metrics['RMSE']:.4f}"
)

print(
    f"R²                 : "
    f"{metrics['R2']:.4f}"
)


# ================================================================
# SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R2"
        ],
        "Value": [
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"]
        ]
    }
)


metrics_file = (
    METRICS_PATH
    / "forecasting_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


# ================================================================
# SAVE VALIDATION PREDICTIONS
# ================================================================

predictions_file = (
    METRICS_PATH
    / "forecasting_validation_predictions.csv"
)


validation_df.to_csv(
    predictions_file,
    index=False
)


# ================================================================
# CREATE ACTUAL VS PREDICTED PLOT
# ================================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    validation_df[
        "Actual QoE"
    ].values,
    label="Actual QoE",
    linewidth=1.5
)

plt.plot(
    validation_df[
        "Predicted QoE"
    ].values,
    label="Predicted QoE",
    linewidth=1.5
)

plt.xlabel(
    "Validation Observation"
)

plt.ylabel(
    "QoE Score"
)

plt.title(
    "Actual vs Predicted QoE - Forecasting Validation"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plot_file = (
    PLOTS_PATH
    / "qoe_forecast_actual_vs_predicted.png"
)

plt.tight_layout()

plt.savefig(
    plot_file,
    dpi=150
)

plt.close()


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print(
    "FILES CREATED"
)

print(
    f"Validation plot:\n"
    f"{plot_file}"
)

print(
    f"Metrics:\n"
    f"{metrics_file}"
)

print(
    f"Validation predictions:\n"
    f"{predictions_file}"
)

print()
print(
    "FORECASTING EVALUATION COMPLETED"
)
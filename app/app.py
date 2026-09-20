"""
5G QoS-to-QoE Prediction Dashboard
-----------------------------------

Streamlit dashboard for:
1. QoE prediction
2. QoE forecasting
3. Forecast validation
4. Explainable AI using SHAP
5. Dataset information
"""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "5g_qos_qoe_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "qoe_gradient_boosting.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "qoe_features.pkl"
VALIDATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "forecasting_validation_predictions.csv"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from feature_engineering import (
    create_network_input,
    get_model_features,
)

from forecasting import (
    forecast_summary_from_current,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="5G QoS & QoE Prediction Dashboard",
    page_icon="5G",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1 {
        margin-bottom: 0.2rem;
    }

    h2 {
        margin-top: 1.2rem;
    }

    h3 {
        margin-top: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_PATH}"
        )

    return joblib.load(FEATURE_PATH)


# ============================================================
# LOAD PROJECT DATA
# ============================================================

try:
    dataset = load_dataset()
    model = load_model()
    model_features = load_features()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_input" not in st.session_state:
    st.session_state.prediction_input = None


# ============================================================
# TITLE
# ============================================================

st.title("5G QoS & QoE Prediction Dashboard")

st.caption(
    "Data-driven QoE prediction, forecasting, validation and explainable AI "
    "using measured 5G QoS parameters."
)


# ============================================================
# SECTION 1 — QoE PREDICTION
# ============================================================

st.header("1. QoE Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    latency = st.number_input(
        "Latency (ms)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
    )

with col2:
    pdv = st.number_input(
        "Packet Delay Variation (ms)",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
    )

with col3:
    throughput_dl = st.number_input(
        "Download Throughput (Mbps)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=1.0,
    )

col4, col5 = st.columns(2)

with col4:
    packet_loss = st.number_input(
        "Packet Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )

with col5:
    throughput_ul = st.number_input(
        "Upload Throughput (Mbps)",
        min_value=0.0,
        max_value=10000.0,
        value=20.0,
        step=1.0,
    )


predict_button = st.button(
    "Predict QoE",
    type="primary",
    width="stretch",
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:
        input_data = create_network_input(
            latency=latency,
            packet_loss=packet_loss,
            pdv=pdv,
            throughput_ul=throughput_ul,
            throughput_dl=throughput_dl,
        )

        input_data = input_data[model_features]

        prediction = float(model.predict(input_data)[0])

        prediction = float(
            np.clip(prediction, 0, 100)
        )

        st.session_state.prediction = prediction

        st.session_state.prediction_input = {
            "latency": latency,
            "packet_loss": packet_loss,
            "pdv": pdv,
            "throughput_ul": throughput_ul,
            "throughput_dl": throughput_dl,
        }

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    if prediction >= 80:
        category = "Excellent"
    elif prediction >= 60:
        category = "Good"
    elif prediction >= 40:
        category = "Fair"
    else:
        category = "Poor"

    st.subheader("Predicted QoE")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "QoE Score",
            f"{prediction:.2f}",
        )

    with metric_col2:
        st.metric(
            "QoE Category",
            category,
        )

    st.info(
        f"The predicted QoE score is {prediction:.2f}, "
        f"which corresponds to the {category} QoE category."
    )


# ============================================================
# SECTION 2 — QoE FORECASTING
# ============================================================

st.header("2. QoE Forecasting")

if st.session_state.prediction is not None:

    try:

        forecast_result = forecast_summary_from_current(
            qoe_series=dataset["qoe"],
            current_qoe=st.session_state.prediction,
            horizon=10,
            n_lags=5,
        )

        current_qoe = forecast_result["current_qoe"]
        future_qoe = forecast_result["future_qoe"]
        change = forecast_result["change"]
        trend = forecast_result["trend"]
        forecast_values = forecast_result["forecast_values"]

        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

        with forecast_col1:
            st.metric(
                "Current QoE",
                f"{current_qoe:.2f}",
            )

        with forecast_col2:
            st.metric(
                "10-Step Forecast",
                f"{future_qoe:.2f}",
            )

        with forecast_col3:
            st.metric(
                "Expected Change",
                f"{change:+.2f}",
            )

        st.write(f"**Forecast Trend:** {trend}")

        historical = dataset["qoe"].tail(20).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7.5, 3))

        historical_x = np.arange(len(historical))

        forecast_x = np.arange(
            len(historical),
            len(historical) + len(forecast_values),
        )

        ax.plot(
            historical_x,
            historical.values,
            marker="o",
            linewidth=1.5,
            label="Historical QoE",
        )

        ax.plot(
            forecast_x,
            forecast_values,
            marker="o",
            linewidth=1.5,
            label="Forecast QoE",
        )

        ax.axvline(
            x=len(historical) - 1,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel("Observation")
        ax.set_ylabel("QoE")
        ax.set_title("QoE Forecast")

        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()

        st.pyplot(
            fig,
            width="content",
        )

        plt.close(fig)

    except Exception as e:
        st.error(f"Forecasting failed: {e}")

else:

    st.info(
        "Enter the QoS parameters and click Predict QoE to generate the forecast."
    )


# ============================================================
# SECTION 3 — FORECAST VALIDATION
# ============================================================

st.header("3. Forecast Validation")

if VALIDATION_PATH.exists():

    try:

        validation_df = pd.read_csv(
            VALIDATION_PATH
        )

        actual = pd.to_numeric(
            validation_df["Actual QoE"],
            errors="coerce",
        )

        predicted = pd.to_numeric(
            validation_df["Predicted QoE"],
            errors="coerce",
        )

        valid_mask = (
            actual.notna()
            & predicted.notna()
        )

        actual = actual[valid_mask]
        predicted = predicted[valid_mask]

        validation_mae = float(
            np.mean(
                np.abs(actual.values - predicted.values)
            )
        )

        validation_rmse = float(
            np.sqrt(
                np.mean(
                    (actual.values - predicted.values) ** 2
                )
            )
        )

        actual_mean = float(
            np.mean(actual.values)
        )

        ss_res = float(
            np.sum(
                (actual.values - predicted.values) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (actual.values - actual_mean) ** 2
            )
        )

        if ss_tot != 0:
            validation_r2 = 1 - (
                ss_res / ss_tot
            )
        else:
            validation_r2 = 0.0

        validation_col1, validation_col2, validation_col3 = st.columns(3)

        with validation_col1:
            st.metric(
                "MAE",
                f"{validation_mae:.4f}",
            )

        with validation_col2:
            st.metric(
                "RMSE",
                f"{validation_rmse:.4f}",
            )

        with validation_col3:
            st.metric(
                "R²",
                f"{validation_r2:.4f}",
            )

        st.write(
            f"**Validation observations:** {len(actual)}"
        )

        # ----------------------------------------------------
        # Current Input Prediction
        # ----------------------------------------------------

        if st.session_state.prediction is not None:

            st.subheader("Current Input Prediction")

            current_col1, current_col2 = st.columns(2)

            with current_col1:
                st.metric(
                    "Current Input QoE",
                    f"{st.session_state.prediction:.2f}",
                )

            with current_col2:
                st.info(
                    "This value updates when the QoS inputs change. "
                    "The historical validation graph below remains fixed "
                    "because it represents previously unseen validation data."
                )

        # ----------------------------------------------------
        # Historical Validation Graph
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(7.5, 3))

        validation_x = np.arange(
            len(actual)
        )

        ax.plot(
            validation_x,
            actual.values,
            linewidth=1.2,
            label="Actual QoE",
        )

        ax.plot(
            validation_x,
            predicted.values,
            linewidth=1.2,
            label="Predicted QoE",
        )

        ax.set_xlabel("Validation Observation")
        ax.set_ylabel("QoE")

        ax.set_title(
            "Historical Actual vs Predicted QoE"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()

        st.pyplot(
            fig,
            width="content",
        )

        plt.close(fig)

        with st.expander("View validation predictions"):

            st.dataframe(
                validation_df,
                width="stretch",
                height=300,
            )

    except Exception as e:

        st.error(
            f"Unable to load validation results: {e}"
        )

else:

    st.warning(
        "Forecast validation file was not found."
    )


# ============================================================
# SECTION 4 — EXPLAINABLE AI
# ============================================================

st.header("4. Explainable AI — SHAP")

if st.session_state.prediction is not None:

    try:

        input_data = create_network_input(
            latency=latency,
            packet_loss=packet_loss,
            pdv=pdv,
            throughput_ul=throughput_ul,
            throughput_dl=throughput_dl,
        )

        input_data = input_data[model_features]

        background = dataset[
            model_features
        ].sample(
            min(30, len(dataset)),
            random_state=42,
        )

        def model_predict(data):

            if isinstance(data, np.ndarray):
                data = pd.DataFrame(
                    data,
                    columns=model_features,
                )

            return model.predict(
                data[model_features]
            )

        explainer = shap.Explainer(
            model_predict,
            background,
        )

        shap_result = explainer(
            input_data
        )

        shap_values = np.asarray(
            shap_result.values
        )

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        feature_names = list(
            model_features
        )

        shap_table = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": shap_values,
                "Absolute Impact": np.abs(
                    shap_values
                ),
            }
        )

        shap_table = shap_table.sort_values(
            "Absolute Impact",
            ascending=False,
        ).reset_index(drop=True)

        st.subheader("Top QoS Factors")

        st.dataframe(
            shap_table.head(10),
            width="stretch",
        )

        # ----------------------------------------------------
        # SHAP Bar Chart
        # ----------------------------------------------------

        top_features = shap_table.head(10).sort_values(
            "Absolute Impact",
            ascending=True,
        )

        fig, ax = plt.subplots(
            figsize=(7.5, 3)
        )

        ax.barh(
            top_features["Feature"],
            top_features["SHAP Value"],
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Feature")

        ax.set_title(
            "SHAP Feature Impact on Current QoE Prediction"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            width="content",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Strongest Feature
        # ----------------------------------------------------

        strongest_feature = shap_table.iloc[0]

        st.subheader("Prediction Explanation")

        st.write(
            f"**Current QoE:** "
            f"{st.session_state.prediction:.2f}"
        )

        st.write(
            f"**Strongest contributing feature:** "
            f"{strongest_feature['Feature']}"
        )

        st.write(
            f"**SHAP impact:** "
            f"{strongest_feature['SHAP Value']:+.4f}"
        )

        if strongest_feature["SHAP Value"] > 0:

            st.info(
                "The strongest feature has a positive contribution "
                "towards the current QoE prediction."
            )

        elif strongest_feature["SHAP Value"] < 0:

            st.info(
                "The strongest feature has a negative contribution "
                "towards the current QoE prediction."
            )

        else:

            st.info(
                "The strongest feature has very little contribution "
                "to the current QoE prediction."
            )

    except Exception as e:

        st.error(
            f"SHAP explanation failed: {e}"
        )

else:

    st.info(
        "Predict QoE first to generate the SHAP explanation."
    )


# ============================================================
# SECTION 5 — DATASET INFORMATION
# ============================================================

st.header("5. Dataset Information")

dataset_col1, dataset_col2, dataset_col3, dataset_col4 = st.columns(4)

with dataset_col1:
    st.metric(
        "Rows",
        f"{len(dataset):,}",
    )

with dataset_col2:
    st.metric(
        "Columns",
        f"{len(dataset.columns):,}",
    )

with dataset_col3:
    st.metric(
        "Model Features",
        f"{len(model_features):,}",
    )

with dataset_col4:
    st.metric(
        "QoE Target",
        "qoe",
    )


st.subheader("Dataset Columns")

st.write(
    list(dataset.columns)
)


st.subheader("Sample Records")

st.dataframe(
    dataset.head(10),
    width="stretch",
)


# ============================================================
# SCENARIO INFORMATION
# ============================================================

if "scenario" in dataset.columns:

    st.subheader("Scenario Distribution")

    scenario_counts = (
        dataset["scenario"]
        .value_counts()
        .reset_index()
    )

    scenario_counts.columns = [
        "Scenario",
        "Observations",
    ]

    st.dataframe(
        scenario_counts,
        width="content",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "5G QoS-to-QoE Prediction | "
    "Gradient Boosting | SHAP | QoE Forecasting"
)
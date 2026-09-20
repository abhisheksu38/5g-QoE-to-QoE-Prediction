"""
Feature engineering utilities for the 5G QoS and QoE prediction project.

The trained model uses 25 QoS features:
- Packet loss statistics
- Latency statistics
- Packet delay variation (PDV) statistics
- Upload throughput statistics
- Download throughput statistics
"""

import pandas as pd


# ============================================================
# EXACT MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "op1_cur_packet_loss_pct_mean",
    "op1_cur_packet_loss_pct_median",
    "op1_cur_packet_loss_pct_min",
    "op1_cur_packet_loss_pct_max",
    "op1_cur_packet_loss_pct_std",

    "op1_cur_latency_median_ms_mean",
    "op1_cur_latency_median_ms_median",
    "op1_cur_latency_median_ms_min",
    "op1_cur_latency_median_ms_max",
    "op1_cur_latency_median_ms_std",

    "op1_cur_pdv_median_ms_mean",
    "op1_cur_pdv_median_ms_median",
    "op1_cur_pdv_median_ms_min",
    "op1_cur_pdv_median_ms_max",
    "op1_cur_pdv_median_ms_std",

    "op1_cur_throughput_ul_mbps_mean",
    "op1_cur_throughput_ul_mbps_median",
    "op1_cur_throughput_ul_mbps_min",
    "op1_cur_throughput_ul_mbps_max",
    "op1_cur_throughput_ul_mbps_std",

    "op1_cur_throughput_dl_mbps_mean",
    "op1_cur_throughput_dl_mbps_median",
    "op1_cur_throughput_dl_mbps_min",
    "op1_cur_throughput_dl_mbps_max",
    "op1_cur_throughput_dl_mbps_std"
]


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features():
    """
    Return the exact 25 features used by the trained model.
    """

    return MODEL_FEATURES.copy()


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(df):
    """
    Check whether all required model features exist.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    bool
        True if all model features are present.

    Raises
    ------
    KeyError
        If one or more features are missing.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise KeyError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    return True


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

def select_model_features(df):
    """
    Select only the 25 features required by the ML model.
    """

    validate_features(df)

    return df[MODEL_FEATURES].copy()


# ============================================================
# CREATE INPUT FROM NETWORK PARAMETERS
# ============================================================

def create_network_input(
    latency,
    packet_loss,
    pdv,
    throughput_ul,
    throughput_dl
):
    """
    Create a single-row dataframe from user-entered
    5G network parameters.

    The same value is used for the mean, median, min,
    max and standard deviation fields because the
    Streamlit application receives a single current
    network measurement.
    """

    input_data = {}

    # --------------------------------------------------------
    # Packet Loss
    # --------------------------------------------------------

    input_data["op1_cur_packet_loss_pct_mean"] = packet_loss
    input_data["op1_cur_packet_loss_pct_median"] = packet_loss
    input_data["op1_cur_packet_loss_pct_min"] = packet_loss
    input_data["op1_cur_packet_loss_pct_max"] = packet_loss
    input_data["op1_cur_packet_loss_pct_std"] = 0.0

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    input_data["op1_cur_latency_median_ms_mean"] = latency
    input_data["op1_cur_latency_median_ms_median"] = latency
    input_data["op1_cur_latency_median_ms_min"] = latency
    input_data["op1_cur_latency_median_ms_max"] = latency
    input_data["op1_cur_latency_median_ms_std"] = 0.0

    # --------------------------------------------------------
    # Packet Delay Variation
    # --------------------------------------------------------

    input_data["op1_cur_pdv_median_ms_mean"] = pdv
    input_data["op1_cur_pdv_median_ms_median"] = pdv
    input_data["op1_cur_pdv_median_ms_min"] = pdv
    input_data["op1_cur_pdv_median_ms_max"] = pdv
    input_data["op1_cur_pdv_median_ms_std"] = 0.0

    # --------------------------------------------------------
    # Upload Throughput
    # --------------------------------------------------------

    input_data["op1_cur_throughput_ul_mbps_mean"] = throughput_ul
    input_data["op1_cur_throughput_ul_mbps_median"] = throughput_ul
    input_data["op1_cur_throughput_ul_mbps_min"] = throughput_ul
    input_data["op1_cur_throughput_ul_mbps_max"] = throughput_ul
    input_data["op1_cur_throughput_ul_mbps_std"] = 0.0

    # --------------------------------------------------------
    # Download Throughput
    # --------------------------------------------------------

    input_data["op1_cur_throughput_dl_mbps_mean"] = throughput_dl
    input_data["op1_cur_throughput_dl_mbps_median"] = throughput_dl
    input_data["op1_cur_throughput_dl_mbps_min"] = throughput_dl
    input_data["op1_cur_throughput_dl_mbps_max"] = throughput_dl
    input_data["op1_cur_throughput_dl_mbps_std"] = 0.0

    return pd.DataFrame(
        [input_data],
        columns=MODEL_FEATURES
    )


# ============================================================
# ADD STATISTICAL FEATURES TO RAW DATA
# ============================================================

def create_statistical_features(
    df,
    source_column,
    prefix
):
    """
    Create mean, median, min, max and standard deviation
    features from a source column.

    Example:
        source_column = latency
        prefix = op1_cur_latency_median_ms
    """

    if source_column not in df.columns:

        raise KeyError(
            f"Column '{source_column}' not found in dataframe."
        )

    result = df.copy()

    result[f"{prefix}_mean"] = result[source_column]
    result[f"{prefix}_median"] = result[source_column]
    result[f"{prefix}_min"] = result[source_column]
    result[f"{prefix}_max"] = result[source_column]
    result[f"{prefix}_std"] = 0.0

    return result
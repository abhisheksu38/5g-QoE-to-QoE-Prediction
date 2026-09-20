"""
Explainable AI (XAI) utilities for the 5G QoE prediction project.

SHAP is used to explain which QoS features have the strongest
influence on the Gradient Boosting QoE prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_explainer(model):
    """
    Create a SHAP TreeExplainer for the trained model.
    """

    return shap.TreeExplainer(model)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(
    model,
    X,
    max_samples=300
):
    """
    Calculate SHAP values for the supplied samples.

    Parameters
    ----------
    model : trained tree-based model
        Gradient Boosting model.

    X : pandas.DataFrame
        Model input features.

    max_samples : int
        Maximum number of samples used for explanation.

    Returns
    -------
    explainer, shap_values, X_sample
    """

    if len(X) > max_samples:

        X_sample = X.sample(
            n=max_samples,
            random_state=42
        )

    else:

        X_sample = X.copy()

    explainer = create_explainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    return (
        explainer,
        shap_values,
        X_sample
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    shap_values,
    X
):
    """
    Calculate mean absolute SHAP importance
    for every feature.
    """

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": importance
        }
    )

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    return result


# ============================================================
# TOP FEATURES
# ============================================================

def get_top_features(
    shap_values,
    X,
    n=10
):
    """
    Return the top n features affecting QoE prediction.
    """

    importance = get_feature_importance(
        shap_values,
        X
    )

    return importance.head(n)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def create_summary_plot(
    shap_values,
    X,
    output_path=None,
    max_display=10
):
    """
    Create a SHAP summary bar plot.
    """

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False
    )

    plt.title(
        "Top Features Affecting QoE Prediction"
    )

    plt.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close()


# ============================================================
# SHAP BEESWARM PLOT
# ============================================================

def create_beeswarm_plot(
    shap_values,
    X,
    output_path=None,
    max_display=10
):
    """
    Create a SHAP beeswarm plot showing
    feature value and prediction impact.
    """

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )

    plt.title(
        "SHAP Impact on QoE Prediction"
    )

    plt.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close()


# ============================================================
# SINGLE PREDICTION EXPLANATION
# ============================================================

def explain_prediction(
    model,
    input_df
):
    """
    Explain a single QoE prediction.

    Returns the SHAP contribution of every feature.
    """

    explainer = create_explainer(
        model
    )

    shap_values = explainer.shap_values(
        input_df
    )

    if isinstance(
        shap_values,
        list
    ):
        shap_values = shap_values[0]

    values = np.asarray(
        shap_values
    )

    if values.ndim == 2:
        values = values[0]

    explanation = pd.DataFrame(
        {
            "feature": input_df.columns,
            "value": input_df.iloc[0].values,
            "shap_impact": values
        }
    )

    explanation["abs_impact"] = (
        explanation["shap_impact"]
        .abs()
    )

    explanation = explanation.sort_values(
        "abs_impact",
        ascending=False
    ).reset_index(drop=True)

    return explanation


# ============================================================
# STRONGEST FEATURE
# ============================================================

def get_strongest_feature(
    explanation
):
    """
    Return the feature with the strongest
    absolute SHAP impact.
    """

    if explanation.empty:

        return None

    return explanation.iloc[0]


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def generate_explanation_text(
    explanation
):
    """
    Generate a simple explanation of the strongest
    QoS factor affecting the prediction.
    """

    if explanation.empty:

        return (
            "No explanation could be generated."
        )

    strongest = explanation.iloc[0]

    feature = strongest["feature"]

    impact = strongest["shap_impact"]

    if impact > 0:

        direction = (
            "increases"
        )

    elif impact < 0:

        direction = (
            "decreases"
        )

    else:

        direction = (
            "has little effect on"
        )

    return (
        f"The strongest model factor is "
        f"{feature}. Its SHAP impact is "
        f"{impact:.3f}, indicating that this "
        f"feature {direction} the predicted QoE."
    )
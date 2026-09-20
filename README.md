# Data-Driven QoS-to-QoE Prediction for 5G Communication Networks

<p align="center">
  <a href="https://5g-qoe-to-qoe-prediction-6hcmlzenuhe5rgqmxqb3h8.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%205G%20QoS--to--QoE%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Demo">
  </a>
</p>

A machine learning based application that predicts Quality of Experience (QoE) from measured 5G Quality of Service (QoS) parameters and provides QoE forecasting and explainable AI analysis.

## Project Overview

This project uses measured network QoS parameters to predict the QoE experienced by a user in a 5G communication environment.

The main idea is:

QoS Parameters → Machine Learning Model → Predicted QoE

The project focuses on eGaming traffic and uses real-world 5G/LTE measurement data from the dataset associated with the paper:

"A Standard-compliant Assessment of Beyond-eMBB QoS/QoE in 5G Networks"

The project uses a Gradient Boosting Regression model for QoE prediction.

## Main Features

### 1. QoE Prediction

The application accepts the following network parameters:

- Latency
- Packet Loss
- Packet Delay Variation (PDV)
- Download Throughput
- Upload Throughput

These parameters are processed into the model feature representation and used to predict QoE.

### 2. QoE Forecasting

The application forecasts future QoE values using a lag-based autoregressive Gradient Boosting model.

The forecasting model uses:

- Previous QoE values
- Lag features
- Rolling mean
- Rolling standard deviation

The dashboard provides a 10-step QoE forecast and identifies the QoE trend.

### 3. Forecast Validation

The forecasting model is evaluated using historical unseen observations.

Validation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The validation dataset contains 620 observations.

### 4. Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain the QoE prediction.

The dashboard shows:

- Important QoS features
- SHAP feature importance
- Feature impact direction
- Strongest contributing feature

This helps understand how individual QoS parameters influence the predicted QoE.

### 5. Dataset Information

The application also provides information about:

- Dataset size
- Number of features
- Model features
- QoE target
- Dataset columns
- Sample records
- Scenario distribution

## Dataset

The project uses the dataset:

"A Standard-compliant Assessment of Beyond-eMBB QoS/QoE in 5G Networks"

Authors:

Giuseppe Caso, Mohammad Rajiullah, Anna Brunstrom, Luca De Nardis, Ozgu Alay, Marco Neri

Published at IEEE CSCN 2024.

Citation:

G. Caso et al., "A Standard-compliant Assessment of Beyond-eMBB QoS/QoE in 5G Networks," IEEE CSCN'24, pp. 1–7, 2024.

The processed dataset used by the application contains 3,440 observations and 25 model input features.

## QoS Features

The model uses statistical representations of:

- Packet Loss
- Latency
- Packet Delay Variation
- Upload Throughput
- Download Throughput

For each QoS parameter, statistical features such as mean, median, minimum, maximum and standard deviation are used.

## Machine Learning Model

### QoE Prediction

Model:

Gradient Boosting Regressor

Target:

QoE

Input:

25 engineered QoS features

The model is stored as:

models/qoe_gradient_boosting.pkl

The corresponding feature list is stored as:

models/qoe_features.pkl

## Forecasting Model

QoE forecasting uses an autoregressive Gradient Boosting model.

The model uses:

- lag_1
- lag_2
- lag_3
- lag_4
- lag_5
- rolling_mean_3
- rolling_mean_5
- rolling_std_5

Source-file-aware chronological validation is used to avoid mixing time-series observations from different measurement files.

## Forecasting Validation

Final source-file-aware validation results:

| Metric | Value |
|---|---:|
| Validation Samples | 620 |
| MAE | 4.8806 |
| RMSE | 9.7766 |
| R² | 0.8227 |

These values represent the historical forecasting validation experiment and are not changed by the live dashboard input.

## Project Structure

```text
5g/
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       └── 5g_qos_qoe_features.csv
│
├── models/
│   ├── qoe_features.pkl
│   └── qoe_gradient_boosting.pkl
│
├── results/
│   ├── metrics/
│   │   ├── forecasting_metrics.csv
│   │   └── forecasting_validation_predictions.csv
│   │
│   └── plots/
│       └── qoe_forecast_actual_vs_predicted.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── train_model.py
│   └── evaluate_forecasting.py
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
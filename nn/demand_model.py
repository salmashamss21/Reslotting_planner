# nn/demand_model.py
# This module builds the supervised learning examples from weekly_demand.csv,
# computes the Last-Week and 4-Week-Average baselines, and trains a simple
# Multi-Layer Perceptron (MLP) neural network to predict next-week units sold.

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# Build supervised learning rows for one product across all weeks starting at week 8.
# Each row uses the past 8 weeks of log(1+units), week_of_year, mean_price, and
# invoice_count to predict that week's actual units sold.
def build_product_examples(prod_df):
    prod_df = prod_df.sort_values(by="week_index").reset_index(drop=True)
    units_list = list(prod_df["units"])
    log_list = list(prod_df["log_units"])
    woy_list = list(prod_df["week_of_year"])
    price_list = list(prod_df["mean_price"])
    inv_list = list(prod_df["invoice_count"])
    split_list = list(prod_df["split"])
    week_list = list(prod_df["week_index"])
    code = str(prod_df["StockCode"].iloc[0])

    records = []
    for idx in range(8, len(prod_df)):
        # Collect the last 8 weeks of log(1 + units) from idx-8 up to idx-1
        features = []
        for lag in range(1, 9):
            features.append(float(log_list[idx - lag]))
        # Add week of year, last week's mean price, and last week's log invoice count
        features.append(float(woy_list[idx]))
        features.append(float(price_list[idx - 1]))
        features.append(float(math.log(1.0 + inv_list[idx - 1])))

        last_week_units = float(units_list[idx - 1])
        four_week_avg = (
            float(units_list[idx - 1]) + float(units_list[idx - 2]) +
            float(units_list[idx - 3]) + float(units_list[idx - 4])
        ) / 4.0

        records.append({
            "StockCode": code,
            "week_index": int(week_list[idx]),
            "split": str(split_list[idx]),
            "features": features,
            "target_units": float(units_list[idx]),
            "target_log_units": float(log_list[idx]),
            "pred_last_week": last_week_units,
            "pred_4week_avg": four_week_avg
        })
    return records


# Convert the full weekly_demand table into a list of supervised examples for all products.
def create_dataset_from_weekly(weekly_df):
    all_records = []
    for code, group in weekly_df.groupby("StockCode"):
        prod_records = build_product_examples(group)
        for rec in prod_records:
            all_records.append(rec)
    return all_records


# Extract numpy arrays (X, y_log, y_units) for a chosen split ('train', 'val', or 'test').
# Keeping splits separate prevents future data from leaking into training.
def extract_split_arrays(all_records, split_name):
    X_rows = []
    y_log_rows = []
    y_unit_rows = []
    matching_records = []

    for rec in all_records:
        if rec["split"] == split_name:
            X_rows.append(rec["features"])
            y_log_rows.append(rec["target_log_units"])
            y_unit_rows.append(rec["target_units"])
            matching_records.append(rec)

    X = np.array(X_rows, dtype=float)
    y_log = np.array(y_log_rows, dtype=float)
    y_units = np.array(y_unit_rows, dtype=float)
    return X, y_log, y_units, matching_records


# Convert log(1 + units) predictions back into regular unit counts.
# We clip values so predictions are never negative or unrealistically huge.
def convert_log_to_units(log_predictions):
    unit_preds = []
    for val in log_predictions:
        clipped_log = max(0.0, min(float(val), 12.0))
        units = math.exp(clipped_log) - 1.0
        unit_preds.append(units)
    return np.array(unit_preds, dtype=float)


# Train a small, easy-to-explain 2-layer neural network (MLP) on the training weeks.
# The StandardScaler is fit ONLY on X_train to obey strict machine learning rules.
def train_neural_network(X_train, y_train_log, seed=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.05,
        learning_rate_init=0.005,
        early_stopping=True,
        validation_fraction=0.12,
        max_iter=300,
        random_state=seed
    )
    model.fit(X_train_scaled, y_train_log)
    return model, scaler


# Predict units sold for a given feature matrix X using the trained MLP and scaler.
def predict_with_nn(model, scaler, X):
    X_scaled = scaler.transform(X)
    pred_log = model.predict(X_scaled)
    pred_units = convert_log_to_units(pred_log)
    return pred_units


# Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in plain loops
def calculate_error_metrics(true_units, predicted_units):
    n = len(true_units)
    if n == 0:
        return 0.0, 0.0

    total_abs_error = 0.0
    total_sq_error = 0.0
    for i in range(n):
        diff = float(true_units[i]) - float(predicted_units[i])
        total_abs_error += abs(diff)
        total_sq_error += (diff * diff)

    mae = total_abs_error / n
    rmse = math.sqrt(total_sq_error / n)
    return round(mae, 2), round(rmse, 2)


# Train the NN, evaluate it against the Last-Week and 4-Week-Average baselines on
# both validation and test sets, and attach predictions to the test records.
def run_demand_forecasting(weekly_df, seed=42):
    all_records = create_dataset_from_weekly(weekly_df)
    X_train, y_train_log, y_train_units, _ = extract_split_arrays(all_records, "train")
    X_val, _, y_val_units, val_records = extract_split_arrays(all_records, "val")
    X_test, _, y_test_units, test_records = extract_split_arrays(all_records, "test")

    model, scaler = train_neural_network(X_train, y_train_log, seed=seed)
    val_nn_preds = predict_with_nn(model, scaler, X_val)
    test_nn_preds = predict_with_nn(model, scaler, X_test)

    val_lw = [r["pred_last_week"] for r in val_records]
    val_4w = [r["pred_4week_avg"] for r in val_records]
    test_lw = [r["pred_last_week"] for r in test_records]
    test_4w = [r["pred_4week_avg"] for r in test_records]

    for i in range(len(test_records)):
        test_records[i]["pred_nn"] = float(test_nn_preds[i])

    metrics = {
        "val_last_week": calculate_error_metrics(y_val_units, val_lw),
        "val_4week_avg": calculate_error_metrics(y_val_units, val_4w),
        "val_nn": calculate_error_metrics(y_val_units, val_nn_preds),
        "test_last_week": calculate_error_metrics(y_test_units, test_lw),
        "test_4week_avg": calculate_error_metrics(y_test_units, test_4w),
        "test_nn": calculate_error_metrics(y_test_units, test_nn_preds),
    }
    return metrics, test_records, model, scaler

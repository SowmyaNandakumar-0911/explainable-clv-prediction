"""
Stage 3 - Baseline Models
Deep-Ensemble CLV Research Project

Trains three baseline models on one-hot encoded features so all baselines
are compared fairly on the same encoding scheme:
  1. Linear Regression   - can't capture non-linear segment/channel effects
  2. Random Forest        - non-linear, but categories treated as independent
  3. Plain XGBoost         - the real baseline your proposed model must beat

Saves the train/test split (by customer_id) so Stages 4-5 use the exact
same split for a fair, leakage-free comparison.

Run from repo root:
    python src/baseline_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "region", "acquisition_channel", "age_group", "preferred_payment_type",
    "customer_segment", "device_type", "top_category",
]
BOOLEAN_COLS = ["email_opt_in", "churned_flag", "is_returning_flag"]
NUMERIC_COLS = [
    "first_purchase_value", "recency_days", "frequency", "monetary_total",
    "monetary_avg", "monetary_std", "avg_discount", "avg_delivery_days",
    "avg_review_score", "review_missing_rate", "category_diversity",
    "tenure_days",
]
TARGET = "true_clv"


def load_and_prepare():
    df = pd.read_csv(f"{PROCESSED_DIR}/model_table.csv")

    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    feature_cols = [c for c in df_encoded.columns
                    if c not in ("customer_id", "signup_date", TARGET)]

    return df, df_encoded, feature_cols


def evaluate(name, y_true, y_pred, results: list):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:35s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  R2={r2:.4f}")
    results.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})


def main():
    df, df_encoded, feature_cols = load_and_prepare()

    train_ids, test_ids = train_test_split(
        df["customer_id"], test_size=0.2, random_state=RANDOM_STATE
    )
    pd.DataFrame({"customer_id": train_ids}).to_csv(
        f"{PROCESSED_DIR}/split_train_ids.csv", index=False)
    pd.DataFrame({"customer_id": test_ids}).to_csv(
        f"{PROCESSED_DIR}/split_test_ids.csv", index=False)

    train_mask = df_encoded["customer_id"].isin(train_ids)
    test_mask = df_encoded["customer_id"].isin(test_ids)

    X_train = df_encoded.loc[train_mask, feature_cols]
    X_test = df_encoded.loc[test_mask, feature_cols]
    y_train = df_encoded.loc[train_mask, TARGET]
    y_test = df_encoded.loc[test_mask, TARGET]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate("Linear Regression", y_test, lr.predict(X_test), results)

    rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", y_test, rf.predict(X_test), results)

    xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    evaluate("Plain XGBoost (one-hot)", y_test, xgb.predict(X_test), results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{PROCESSED_DIR}/baseline_results.csv", index=False)
    print(f"\nSaved baseline results to {PROCESSED_DIR}/baseline_results.csv")
    print("(Stage 5 will append the proposed embedding+XGBoost model to this table)")


if __name__ == "__main__":
    main()
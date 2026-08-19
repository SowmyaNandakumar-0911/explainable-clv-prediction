"""
Stage 5 - Final Model: Entity Embeddings + XGBoost
Deep-Ensemble CLV Research Project

Trains the proposed model: XGBoost on top of the entity-embedding features
produced in Stage 4 (learned category vectors + engineered numeric
features), instead of one-hot encoded categories. Appends the result to
baseline_results.csv for a direct, same-metric comparison.

Run from repo root:
    python src/train_gbm.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42
TARGET = "true_clv"


def main():
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(f"{PROCESSED_DIR}/embedded_features.csv")
    train_ids = pd.read_csv(f"{PROCESSED_DIR}/split_train_ids.csv")["customer_id"]
    test_ids = pd.read_csv(f"{PROCESSED_DIR}/split_test_ids.csv")["customer_id"]

    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]

    train_mask = df["customer_id"].isin(train_ids)
    test_mask = df["customer_id"].isin(test_ids)

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, TARGET]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, TARGET]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")

    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{'Embedding + XGBoost (proposed)':35s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  R2={r2:.4f}")

    # append to the baseline comparison table
    results_df = pd.read_csv(f"{PROCESSED_DIR}/baseline_results.csv")
    new_row = pd.DataFrame([{
        "model": "Embedding + XGBoost (proposed)", "mae": mae, "rmse": rmse, "r2": r2
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(f"{PROCESSED_DIR}/baseline_results.csv", index=False)

    print("\n--- Full comparison table ---")
    print(results_df.to_string(index=False))

    with open(f"{MODELS_DIR}/final_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved trained model to {MODELS_DIR}/final_xgb_model.pkl")
    print(f"Updated comparison table saved to {PROCESSED_DIR}/baseline_results.csv")


if __name__ == "__main__":
    main()
"""
Phase 2B - Feature Group Experiment
Deep-Ensemble CLV Research Project

Tests which groups of advanced features improve CLV prediction.

Uses the SAME train/test customer split as the existing pipeline
to ensure a fair comparison.

Run from repo root:
    python src/feature_experiment.py
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42
TARGET = "true_clv"


def train_and_evaluate(df, feature_cols, train_ids, test_ids, name):

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    train_mask = df["customer_id"].isin(train_ids)
    test_mask = df["customer_id"].isin(test_ids)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    categorical_cols = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_cols = [
        c for c in X_train.columns
        if c not in categorical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]),
                numeric_cols
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical_cols
            )
        ]
    )

    model = XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print(
        f"{name:35s} "
        f"R2={r2:.4f}  "
        f"MAE={mae:8.2f}  "
        f"RMSE={rmse:8.2f}"
    )

    return {
        "feature_set": name,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_features": len(feature_cols)
    }


def main():

    print("Loading advanced feature table...")

    df = pd.read_csv(
        f"{PROCESSED_DIR}/model_table_v2.csv"
    )

    train_ids = pd.read_csv(
        f"{PROCESSED_DIR}/split_train_ids.csv"
    )["customer_id"]

    test_ids = pd.read_csv(
        f"{PROCESSED_DIR}/split_test_ids.csv"
    )["customer_id"]

    print(
        f"Train customers: {len(train_ids)}"
    )

    print(
        f"Test customers: {len(test_ids)}"
    )

    # ---------------------------------------------------------
    # BASIC FEATURES
    # ---------------------------------------------------------

    basic_features = [
        "region",
        "acquisition_channel",
        "age_group",
        "preferred_payment_type",
        "customer_segment",
        "device_type",
        "email_opt_in",
        "churned_flag",
        "first_purchase_value",
        "recency_days",
        "frequency",
        "monetary_total",
        "monetary_avg",
        "monetary_std",
        "avg_discount",
        "avg_delivery_days",
        "avg_review_score",
        "review_missing_rate",
        "category_diversity",
        "tenure_days",
        "top_category"
    ]

    # ---------------------------------------------------------
    # MONETARY FEATURES
    # ---------------------------------------------------------

    monetary_features = [
        "monetary_median",
        "monetary_min",
        "monetary_max",
        "monetary_q25",
        "monetary_q75",
        "total_quantity",
        "avg_quantity",
        "max_quantity",
        "quantity_std",
        "log_monetary_total",
        "log_monetary_avg",
        "log_monetary_max",
        "log_monetary_median",
        "log_total_quantity",
        "log_monetary_q75",
        "max_to_avg_order_ratio",
        "median_to_avg_order_ratio",
        "quantity_per_order"
    ]

    # ---------------------------------------------------------
    # TEMPORAL FEATURES
    # ---------------------------------------------------------

    temporal_features = [
        "purchase_span_days",
        "active_months",
        "avg_days_between_orders",
        "median_days_between_orders",
        "std_days_between_orders",
        "orders_last_30d",
        "spend_last_30d",
        "avg_order_last_30d",
        "quantity_last_30d",
        "orders_last_60d",
        "spend_last_60d",
        "avg_order_last_60d",
        "quantity_last_60d",
        "orders_last_90d",
        "spend_last_90d",
        "avg_order_last_90d",
        "quantity_last_90d",
        "orders_last_180d",
        "spend_last_180d",
        "avg_order_last_180d",
        "quantity_last_180d"
    ]

    # ---------------------------------------------------------
    # BEHAVIORAL FEATURES
    # ---------------------------------------------------------

    behavioral_features = [
        "spend_trend_90d",
        "frequency_trend_90d",
        "aov_recent_vs_overall",
        "spend_per_active_month",
        "orders_per_active_month",
        "category_entropy",
        "top_category_share",
        "discounted_order_rate",
        "discount_std"
    ]

    # ---------------------------------------------------------
    # PAYMENT FEATURES
    # ---------------------------------------------------------

    payment_features = [
        "payment_rate_completed",
        "payment_rate_failed",
        "payment_rate_pending",
        "payment_rate_refunded"
    ]

    results = []

    # ---------------------------------------------------------
    # EXPERIMENT 1
    # ---------------------------------------------------------

    results.append(
        train_and_evaluate(
            df,
            basic_features,
            train_ids,
            test_ids,
            "Basic RFM + customer attributes"
        )
    )

    # ---------------------------------------------------------
    # EXPERIMENT 2
    # ---------------------------------------------------------

    features = basic_features + monetary_features

    results.append(
        train_and_evaluate(
            df,
            features,
            train_ids,
            test_ids,
            "Basic + monetary distribution"
        )
    )

    # ---------------------------------------------------------
    # EXPERIMENT 3
    # ---------------------------------------------------------

    features = (
        basic_features
        + monetary_features
        + temporal_features
    )

    results.append(
        train_and_evaluate(
            df,
            features,
            train_ids,
            test_ids,
            "Basic + monetary + temporal"
        )
    )

    # ---------------------------------------------------------
    # EXPERIMENT 4
    # ---------------------------------------------------------

    features = (
        basic_features
        + monetary_features
        + temporal_features
        + behavioral_features
    )

    results.append(
        train_and_evaluate(
            df,
            features,
            train_ids,
            test_ids,
            "Basic + monetary + temporal + behavioral"
        )
    )

    # ---------------------------------------------------------
    # EXPERIMENT 5
    # ---------------------------------------------------------

    features = (
        basic_features
        + monetary_features
        + temporal_features
        + behavioral_features
        + payment_features
    )

    results.append(
        train_and_evaluate(
            df,
            features,
            train_ids,
            test_ids,
            "ALL advanced features"
        )
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "r2",
        ascending=False
    )

    print("\n")
    print("=" * 80)
    print("FEATURE GROUP EXPERIMENT RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(index=False)
    )

    output_path = (
        f"{PROCESSED_DIR}/feature_experiment_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved results to: {output_path}"
    )


if __name__ == "__main__":
    main()
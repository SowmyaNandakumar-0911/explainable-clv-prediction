"""
Stage 2 - Feature Engineering
Deep-Ensemble CLV Research Project

Builds RFM + behavioral features per customer from cleaned transactions,
merges with customer attributes, and produces a single model-ready table.
Deliberately excludes total_orders (leakage risk identified in Stage 1).

Run from repo root:
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"


def build_rfm_features(transactions: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = transactions["order_date"].max()

    grouped = transactions.groupby("customer_id")

    rfm = grouped.agg(
        recency_days=("order_date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary_total=("order_value", "sum"),
        monetary_avg=("order_value", "mean"),
        monetary_std=("order_value", "std"),
        avg_discount=("discount_applied", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_review_score=("review_score", "mean"),
        review_missing_rate=("review_score_missing_flag", "mean"),
        category_diversity=("product_category", "nunique"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    ).reset_index()

    # std is NaN for single-transaction customers -- fill with 0
    rfm["monetary_std"] = rfm["monetary_std"].fillna(0)
    rfm["avg_review_score"] = rfm["avg_review_score"].fillna(rfm["avg_review_score"].median())

    # customer's most frequently ordered category -- useful embedding-adjacent
    # signal, kept as a plain categorical feature here
    top_category = (
        transactions.groupby(["customer_id", "product_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["customer_id", "count"], ascending=[True, False])
        .drop_duplicates("customer_id")[["customer_id", "product_category"]]
        .rename(columns={"product_category": "top_category"})
    )
    rfm = rfm.merge(top_category, on="customer_id", how="left")

    return rfm


def build_model_table(customers: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(rfm, on="customer_id", how="left")

    # customers with zero post-cleaning transactions (rare, but possible
    # after dropping bad rows) -- fill RFM fields with "no activity" values
    no_activity = df["frequency"].isna()
    print(f"Customers with zero valid transactions after cleaning: {no_activity.sum()}")

    df["recency_days"] = df["recency_days"].fillna(-1)  # -1 = no purchases
    df["frequency"] = df["frequency"].fillna(0)
    for col in ["monetary_total", "monetary_avg", "monetary_std", "avg_discount",
                "avg_delivery_days", "avg_review_score", "review_missing_rate",
                "category_diversity"]:
        df[col] = df[col].fillna(0)
    df["top_category"] = df["top_category"].fillna("none")

    # tenure feature
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    max_date = df["last_order_date"].max()
    df["tenure_days"] = (max_date - df["signup_date"]).dt.days

    # explicitly drop the leakage-risk column identified in Stage 1
    df = df.drop(columns=["total_orders", "total_orders_flag_leakage_risk",
                           "is_returning_flag", "first_order_date", "last_order_date"])
    df = df.rename(columns={"is_returning_flag_cleaned": "is_returning_flag"})

    return df


def main():
    customers = pd.read_csv(f"{PROCESSED_DIR}/customers_clean.csv")
    transactions = pd.read_csv(f"{PROCESSED_DIR}/transactions_clean.csv")
    transactions["order_date"] = pd.to_datetime(transactions["order_date"])

    rfm = build_rfm_features(transactions)
    model_table = build_model_table(customers, rfm)

    print("\n--- Feature engineering summary ---")
    print(f"Final model table shape: {model_table.shape}")
    print(f"Columns: {list(model_table.columns)}")

    model_table.to_csv(f"{PROCESSED_DIR}/model_table.csv", index=False)
    print(f"\nSaved model-ready table to {PROCESSED_DIR}/model_table.csv")


if __name__ == "__main__":
    main()
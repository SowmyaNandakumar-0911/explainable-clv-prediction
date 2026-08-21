"""
Stage 2.1 - Advanced Feature Engineering
Deep-Ensemble CLV Research Project

Adds richer customer behavioral, temporal, monetary and product features
on top of the original RFM features.

This is an experimental improvement over Stage 2.
The original model_table.csv is NOT modified.

Output:
    data/processed/model_table_v2.csv

Run from repo root:
    python src/advanced_features.py
"""

import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"


def build_advanced_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:

    transactions = transactions.copy()

    transactions["order_date"] = pd.to_datetime(transactions["order_date"])

    # Normalize payment-status variants
    transactions["payment_status"] = (
        transactions["payment_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"complete": "completed"})
    )

    reference_date = transactions["order_date"].max()

    grouped = transactions.groupby("customer_id")

    # ---------------------------------------------------------
    # 1. BASIC RFM + MONETARY DISTRIBUTION
    # ---------------------------------------------------------

    features = grouped.agg(
        recency_days=(
            "order_date",
            lambda x: (reference_date - x.max()).days
        ),
        frequency=("transaction_id", "count"),
        monetary_total=("order_value", "sum"),
        monetary_avg=("order_value", "mean"),
        monetary_std=("order_value", "std"),
        monetary_median=("order_value", "median"),
        monetary_min=("order_value", "min"),
        monetary_max=("order_value", "max"),
        monetary_q25=("order_value", lambda x: x.quantile(0.25)),
        monetary_q75=("order_value", lambda x: x.quantile(0.75)),
        total_quantity=("quantity", "sum"),
        avg_quantity=("quantity", "mean"),
        max_quantity=("quantity", "max"),
        quantity_std=("quantity", "std"),
        avg_discount=("discount_applied", "mean"),
        max_discount=("discount_applied", "max"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_review_score=("review_score", "mean"),
        review_missing_rate=("review_score_missing_flag", "mean"),
        category_diversity=("product_category", "nunique"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    ).reset_index()

    # ---------------------------------------------------------
    # 2. PURCHASE SPAN / CUSTOMER ACTIVITY
    # ---------------------------------------------------------

    features["purchase_span_days"] = (
        features["last_order_date"] -
        features["first_order_date"]
    ).dt.days

    features["purchase_span_days"] = features["purchase_span_days"].fillna(0)

    features["active_months"] = (
        features["purchase_span_days"] / 30.44
    ).clip(lower=0)

    # ---------------------------------------------------------
    # 3. PURCHASE INTERVAL FEATURES
    # ---------------------------------------------------------

    sorted_tx = transactions.sort_values(
        ["customer_id", "order_date"]
    ).copy()

    sorted_tx["days_since_previous_order"] = (
        sorted_tx.groupby("customer_id")["order_date"]
        .diff()
        .dt.days
    )

    interval_features = sorted_tx.groupby("customer_id").agg(
        avg_days_between_orders=(
            "days_since_previous_order",
            "mean"
        ),
        median_days_between_orders=(
            "days_since_previous_order",
            "median"
        ),
        std_days_between_orders=(
            "days_since_previous_order",
            "std"
        ),
    ).reset_index()

    features = features.merge(
        interval_features,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 4. RECENT BEHAVIOR
    # ---------------------------------------------------------

    recent_features = []

    for days in [30, 60, 90, 180]:

        cutoff = reference_date - pd.Timedelta(days=days)

        recent = transactions[
            transactions["order_date"] >= cutoff
        ]

        recent_grouped = recent.groupby("customer_id").agg(
            **{
                f"orders_last_{days}d": (
                    "transaction_id",
                    "count"
                ),
                f"spend_last_{days}d": (
                    "order_value",
                    "sum"
                ),
                f"avg_order_last_{days}d": (
                    "order_value",
                    "mean"
                ),
                f"quantity_last_{days}d": (
                    "quantity",
                    "sum"
                ),
            }
        ).reset_index()

        recent_features.append(recent_grouped)

    for recent_df in recent_features:
        features = features.merge(
            recent_df,
            on="customer_id",
            how="left"
        )

    # ---------------------------------------------------------
    # 5. RECENT VS HISTORICAL TREND
    # ---------------------------------------------------------

    features["spend_trend_90d"] = (
        features["spend_last_90d"].fillna(0)
        -
        (
            features["monetary_total"].fillna(0)
            -
            features["spend_last_90d"].fillna(0)
        )
        / 3
    )

    features["frequency_trend_90d"] = (
        features["orders_last_90d"].fillna(0)
        -
        (
            features["frequency"].fillna(0)
            -
            features["orders_last_90d"].fillna(0)
        )
        / 3
    )

    features["aov_recent_vs_overall"] = (
        features["avg_order_last_90d"]
        /
        features["monetary_avg"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # 6. CATEGORY BEHAVIOR
    # ---------------------------------------------------------

    category_counts = (
        transactions
        .groupby(["customer_id", "product_category"])
        .size()
        .reset_index(name="category_orders")
    )

    category_counts["customer_total_orders"] = (
        category_counts
        .groupby("customer_id")["category_orders"]
        .transform("sum")
    )

    category_counts["category_share"] = (
        category_counts["category_orders"]
        /
        category_counts["customer_total_orders"]
    )

    top_category_share = (
        category_counts
        .groupby("customer_id")["category_share"]
        .max()
        .reset_index(name="top_category_share")
    )

    top_category = (
        category_counts
        .sort_values(
            ["customer_id", "category_orders"],
            ascending=[True, False]
        )
        .drop_duplicates("customer_id")
        [["customer_id", "product_category"]]
        .rename(columns={"product_category": "top_category"})
    )

    features = features.merge(
        top_category_share,
        on="customer_id",
        how="left"
    )

    features = features.merge(
        top_category,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 7. CATEGORY CONCENTRATION / ENTROPY
    # ---------------------------------------------------------

    category_counts["log_share"] = np.log(
        category_counts["category_share"].clip(lower=1e-10)
    )

    entropy = (
        category_counts
        .assign(
            entropy_component=lambda x:
            -x["category_share"] * x["log_share"]
        )
        .groupby("customer_id")["entropy_component"]
        .sum()
        .reset_index(name="category_entropy")
    )

    features = features.merge(
        entropy,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 8. DISCOUNT BEHAVIOR
    # ---------------------------------------------------------

    discount_behavior = transactions.groupby("customer_id").agg(
        discounted_order_rate=(
            "discount_applied",
            lambda x: (x > 0).mean()
        ),
        discount_std=("discount_applied", "std"),
    ).reset_index()

    features = features.merge(
        discount_behavior,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 9. PAYMENT BEHAVIOR
    # ---------------------------------------------------------

    payment_pivot = pd.crosstab(
        transactions["customer_id"],
        transactions["payment_status"],
        normalize="index"
    ).reset_index()

    payment_pivot.columns = [
        str(col) if col == "customer_id"
        else f"payment_rate_{col}"
        for col in payment_pivot.columns
    ]

    features = features.merge(
        payment_pivot,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 10. LOG-TRANSFORMED MONETARY FEATURES
    # ---------------------------------------------------------

    for col in [
        "monetary_total",
        "monetary_avg",
        "monetary_max",
        "monetary_median",
        "total_quantity",
        "monetary_q75",
    ]:
        features[f"log_{col}"] = np.log1p(
            features[col].clip(lower=0)
        )

    # ---------------------------------------------------------
    # 11. CUSTOMER SPENDING INTENSITY
    # ---------------------------------------------------------

    features["spend_per_active_month"] = (
        features["monetary_total"]
        /
        features["active_months"].clip(lower=1 / 30.44)
    )

    features["orders_per_active_month"] = (
        features["frequency"]
        /
        features["active_months"].clip(lower=1 / 30.44)
    )

    # ---------------------------------------------------------
    # 12. AOV AND QUANTITY RATIOS
    # ---------------------------------------------------------

    features["max_to_avg_order_ratio"] = (
        features["monetary_max"]
        /
        features["monetary_avg"].replace(0, np.nan)
    )

    features["median_to_avg_order_ratio"] = (
        features["monetary_median"]
        /
        features["monetary_avg"].replace(0, np.nan)
    )

    features["quantity_per_order"] = (
        features["total_quantity"]
        /
        features["frequency"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # 13. CLEAN UP MISSING VALUES
    # ---------------------------------------------------------

    numeric_cols = features.select_dtypes(
        include=[np.number]
    ).columns

    features[numeric_cols] = features[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    features[numeric_cols] = features[numeric_cols].fillna(0)

    # ---------------------------------------------------------
    # 14. SORT FOR CONSISTENCY
    # ---------------------------------------------------------

    features = features.sort_values("customer_id").reset_index(drop=True)

    return features


def main():

    print("Loading cleaned data...")

    customers = pd.read_csv(
        f"{PROCESSED_DIR}/customers_clean.csv"
    )

    transactions = pd.read_csv(
        f"{PROCESSED_DIR}/transactions_clean.csv"
    )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    print(f"Customers: {len(customers)}")
    print(f"Transactions: {len(transactions)}")

    advanced = build_advanced_transaction_features(
        transactions
    )

    # ---------------------------------------------------------
    # Merge customer-level attributes
    # ---------------------------------------------------------

    model_table = customers.merge(
        advanced,
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Tenure
    # ---------------------------------------------------------

    max_order_date = pd.to_datetime(
        transactions["order_date"]
    ).max()

    model_table["tenure_days"] = (
        max_order_date -
        model_table["signup_date"]
    ).dt.days

    # ---------------------------------------------------------
    # Remove leakage-risk columns
    # ---------------------------------------------------------

    drop_cols = [
        "total_orders",
        "total_orders_flag_leakage_risk",
        "is_returning_flag",
        "is_returning_flag_cleaned",
        "signup_date",
        "first_order_date",
        "last_order_date",
    ]

    existing_drop_cols = [
        col for col in drop_cols
        if col in model_table.columns
    ]

    model_table = model_table.drop(
        columns=existing_drop_cols
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path = (
        f"{PROCESSED_DIR}/model_table_v2.csv"
    )

    model_table.to_csv(
        output_path,
        index=False
    )

    print("\n--- Advanced Feature Engineering Summary ---")
    print(
        f"Final shape: {model_table.shape}"
    )

    print(
        f"Number of features excluding ID/target: "
        f"{len(model_table.columns) - 2}"
    )

    print("\nNew behavioral features include:")
    print("  - monetary distribution statistics")
    print("  - purchase interval statistics")
    print("  - purchase span and active months")
    print("  - 30/60/90/180-day recent behavior")
    print("  - spending and frequency trends")
    print("  - category concentration and entropy")
    print("  - discount behavior")
    print("  - payment behavior")
    print("  - quantity behavior")
    print("  - log-transformed monetary features")
    print("  - spending intensity")

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()
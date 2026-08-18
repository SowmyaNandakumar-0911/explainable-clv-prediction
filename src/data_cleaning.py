"""
Stage 1 - Data Cleaning
Deep-Ensemble CLV Research Project

Cleans customers.csv and transactions.csv, resolving every intentional
data-quality issue documented in data_dictionary.csv, and writes cleaned
versions to data/processed/.

Run from repo root:
    python src/data_cleaning.py
"""

import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def clean_customers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Standardize customer_segment casing/whitespace
    df["customer_segment"] = (
        df["customer_segment"].str.strip().str.title()
    )

    # 2. age_group: keep missing as an explicit "Unknown" category rather
    #    than dropping rows or imputing a guessed age band
    df["age_group"] = df["age_group"].fillna("Unknown")

    # 3. churned_flag: ~3% of labels were intentionally flipped in this
    #    synthetic set. We cannot "un-flip" noise we don't know the ground
    #    truth for -- document as a known label-noise limitation instead
    #    of silently correcting it.

    # 4. is_returning_flag inconsistency vs total_orders: recompute this
    #    flag directly rather than trusting the stored value
    df["is_returning_flag_cleaned"] = df["total_orders"] > 1

    # 5. total_orders: flagged as a leakage risk (correlates ~0.97 with
    #    actual transaction count). We keep the raw column for reference
    #    but will exclude it from model features in Stage 2.
    df["total_orders_flag_leakage_risk"] = True

    df["signup_date"] = pd.to_datetime(df["signup_date"])

    return df


def clean_transactions(path: str, valid_customer_ids: set) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Drop full-row duplicates (~1.5% of rows)
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate transaction rows")

    # 2. Drop transactions referencing unknown customers (defensive check)
    df = df[df["customer_id"].isin(valid_customer_ids)]

    # 3. Standardize payment_status casing/whitespace
    df["payment_status"] = df["payment_status"].str.strip().str.lower()

    # 4. order_value: negative values are data errors (refunds recorded
    #    as raw order rows) -- flag and remove rather than silently
    #    flipping sign, since we can't distinguish real refunds from
    #    entry errors
    n_negative_orders = (df["order_value"] < 0).sum()
    df = df[df["order_value"] >= 0]

    # 5. quantity == 0 is a data entry error -- these rows have no real
    #    purchase, drop them
    n_zero_qty = (df["quantity"] == 0).sum()
    df = df[df["quantity"] > 0]

    # 6. delivery_days: negative or absurd (>180 day) values are entry
    #    errors -- cap rather than drop, to preserve the transaction's
    #    order_value/category info
    df["delivery_days_outlier_flag"] = (
        (df["delivery_days"] < 0) | (df["delivery_days"] > 180)
    )
    median_delivery = df.loc[~df["delivery_days_outlier_flag"], "delivery_days"].median()
    df.loc[df["delivery_days_outlier_flag"], "delivery_days"] = median_delivery

    # 7. review_score: ~15% missing -- keep as NaN, do NOT impute with
    #    mean/median, since "no review left" is itself informative and
    #    will be handled explicitly as a missingness feature in Stage 2
    df["review_score_missing_flag"] = df["review_score"].isna()

    df["order_date"] = pd.to_datetime(df["order_date"])

    print(f"Removed {n_negative_orders} negative order_value rows")
    print(f"Removed {n_zero_qty} zero-quantity rows")
    print(f"Capped {df['delivery_days_outlier_flag'].sum()} delivery_days outliers")
    print(f"Review scores missing (kept as NaN + flagged): {df['review_score_missing_flag'].sum()}")

    return df


def main():
    import os

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    customers = clean_customers(f"{RAW_DIR}/customers.csv")
    transactions = clean_transactions(
        f"{RAW_DIR}/transactions.csv", set(customers["customer_id"])
    )

    print("\n--- Cleaning summary ---")
    print(f"Customers: {len(customers)} rows retained")
    print(f"Transactions: {len(transactions)} rows retained")

    customers.to_csv(f"{PROCESSED_DIR}/customers_clean.csv", index=False)
    transactions.to_csv(f"{PROCESSED_DIR}/transactions_clean.csv", index=False)
    print(f"\nSaved cleaned files to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
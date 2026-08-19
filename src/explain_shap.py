"""
Stage 6 - SHAP Explainability
Deep-Ensemble CLV Research Project

Runs SHAP on the trained embedding+XGBoost model. The tricky part: SHAP
gives one value per raw feature, but our categorical variables are now
spread across multiple embedding dimensions (e.g. region_emb_0, _1, _2).
A SHAP value on a single embedding dimension isn't interpretable to a
human on its own -- so we aggregate the SHAP values across each
categorical variable's embedding block back into ONE importance score
per original category column. This is the "traced back through
embeddings" explainability step that's the core contribution of your
paper.

Run from repo root:
    python src/explain_shap.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
TARGET = "true_clv"

CATEGORICAL_COLS = [
    "region", "acquisition_channel", "age_group", "preferred_payment_type",
    "customer_segment", "device_type", "top_category",
]


def main():
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(f"{PROCESSED_DIR}/embedded_features.csv")
    test_ids = pd.read_csv(f"{PROCESSED_DIR}/split_test_ids.csv")["customer_id"]
    test_mask = df["customer_id"].isin(test_ids)

    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]
    X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)

    with open(f"{MODELS_DIR}/final_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

    # ---- Step 1: raw per-feature global importance (mean |SHAP|) ----
    raw_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    raw_importance.to_csv(f"{PROCESSED_DIR}/shap_raw_feature_importance.csv")

    # ---- Step 2: aggregate embedding-block SHAP values back into ONE
    #      importance score per original categorical column ----
    aggregated = {}
    for col in CATEGORICAL_COLS:
        emb_cols = [c for c in feature_cols if c.startswith(f"{col}_emb_")]
        if emb_cols:
            idx = [feature_cols.index(c) for c in emb_cols]
            # sum of |SHAP| across all dims of this category's embedding
            aggregated[col] = np.abs(shap_values[:, idx]).sum(axis=1).mean()

    # non-embedding (plain numeric/boolean) features keep their own SHAP value
    numeric_like_cols = [c for c in feature_cols
                         if not any(c.startswith(f"{cc}_emb_") for cc in CATEGORICAL_COLS)]
    for col in numeric_like_cols:
        aggregated[col] = raw_importance[col]

    aggregated_importance = pd.Series(aggregated).sort_values(ascending=False)
    aggregated_importance.to_csv(f"{PROCESSED_DIR}/shap_aggregated_importance.csv")

    print("\n--- Aggregated global feature importance (embeddings collapsed) ---")
    print(aggregated_importance.to_string())

    # ---- Plot: aggregated global importance bar chart ----
    plt.figure(figsize=(9, 7))
    aggregated_importance.head(15).sort_values().plot(kind="barh")
    plt.xlabel("Mean |SHAP value| (aggregated across embedding dims where applicable)")
    plt.title("Global Feature Importance: Deep-Ensemble CLV Model")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_aggregated_importance.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {FIGURES_DIR}/shap_aggregated_importance.png")

    # ---- Local explanation: pick 2 example customers and show their
    #      top contributing (aggregated) features ----
    print("\n--- Local explanation examples ---")
    for row_idx in [0, 1]:
        customer_id = df.loc[test_mask, "customer_id"].reset_index(drop=True)[row_idx]
        predicted = model.predict(X_test.iloc[[row_idx]])[0]
        actual = df.loc[test_mask, TARGET].reset_index(drop=True)[row_idx]

        local_agg = {}
        for col in CATEGORICAL_COLS:
            emb_cols = [c for c in feature_cols if c.startswith(f"{col}_emb_")]
            if emb_cols:
                idx = [feature_cols.index(c) for c in emb_cols]
                local_agg[col] = shap_values[row_idx, idx].sum()
        for col in numeric_like_cols:
            local_agg[col] = shap_values[row_idx, feature_cols.index(col)]

        local_series = pd.Series(local_agg).sort_values(key=abs, ascending=False)
        print(f"\nCustomer {customer_id}: predicted={predicted:.0f}, actual={actual:.0f}")
        print("Top 5 contributing features (signed, +/- pushes prediction up/down):")
        print(local_series.head(5).to_string())

    print(f"\nRaw per-dimension SHAP saved to {PROCESSED_DIR}/shap_raw_feature_importance.csv")
    print(f"Category-level aggregated SHAP saved to {PROCESSED_DIR}/shap_aggregated_importance.csv")


if __name__ == "__main__":
    main()
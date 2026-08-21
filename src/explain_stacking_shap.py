"""
Stage 9 - SHAP for the Deep-Ensemble
Deep-Ensemble CLV Research Project

A stacked ensemble isn't a single tree model, so SHAP can't be applied to
it directly the naive way. The clean, mathematically correct approach:
since the meta-learner is a LINEAR (Ridge) combination of the base
models' predictions, and SHAP values are additive, the ensemble's SHAP
value for any feature is simply the meta-learner-weighted SUM of that
feature's SHAP value across each tree-based base model:

    SHAP_ensemble(feature) = sum_i( meta_weight_i * SHAP_base_model_i(feature) )

This is exact for the 3 boosting base models (they're purely additive
trees) and a good approximation for Random Forest. We compute it for all
four, then aggregate embedding-block SHAP values back to original
category names, exactly as in Stage 6.

Run from repo root:
    python src/explain_stacking_shap.py
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
TARGET = "true_clv"

CATEGORICAL_COLS = [
    "region", "acquisition_channel", "age_group", "preferred_payment_type",
    "customer_segment", "device_type", "top_category", "segment_channel",
]


def main():
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(f"{PROCESSED_DIR}/embedded_features.csv")
    test_ids = pd.read_csv(f"{PROCESSED_DIR}/split_test_ids.csv")["customer_id"]
    test_mask = df["customer_id"].isin(test_ids)

    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]
    X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)

    with open(f"{MODELS_DIR}/stacking_base_models.pkl", "rb") as f:
        base_models = pickle.load(f)
    with open(f"{MODELS_DIR}/stacking_meta_model.pkl", "rb") as f:
        meta = pickle.load(f)

    meta_weights = dict(zip(base_models.keys(), meta.coef_))
    print(f"Meta-learner weights used to combine SHAP values: {meta_weights}\n")

    # weighted sum of SHAP values across base models
    print("Computing SHAP values for each base learner...")
    ensemble_shap = np.zeros((len(X_test), len(feature_cols)))
    for name, model in base_models.items():
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        ensemble_shap += meta_weights[name] * sv
        print(f"  {name} SHAP computed, weight={meta_weights[name]:.4f}")

    # ---- aggregate embedding-block SHAP back to original category columns ----
    raw_importance = pd.Series(
        np.abs(ensemble_shap).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    aggregated = {}
    for col in CATEGORICAL_COLS:
        emb_cols = [c for c in feature_cols if c.startswith(f"{col}_emb_")]
        if emb_cols:
            idx = [feature_cols.index(c) for c in emb_cols]
            aggregated[col] = np.abs(ensemble_shap[:, idx]).sum(axis=1).mean()

    numeric_like_cols = [c for c in feature_cols
                         if not any(c.startswith(f"{cc}_emb_") for cc in CATEGORICAL_COLS)]
    for col in numeric_like_cols:
        aggregated[col] = raw_importance[col]

    aggregated_importance = pd.Series(aggregated).sort_values(ascending=False)
    aggregated_importance.to_csv(f"{PROCESSED_DIR}/shap_ensemble_aggregated_importance.csv")

    print("\n--- Deep-Ensemble aggregated global feature importance ---")
    print(aggregated_importance.to_string())

    plt.figure(figsize=(9, 7))
    aggregated_importance.head(15).sort_values().plot(kind="barh", color="#2b6cb0")
    plt.xlabel("Meta-weighted mean |SHAP value| (aggregated across embedding dims)")
    plt.title("Global Feature Importance: Deep-Ensemble CLV Model")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_ensemble_aggregated_importance.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {FIGURES_DIR}/shap_ensemble_aggregated_importance.png")


if __name__ == "__main__":
    main()
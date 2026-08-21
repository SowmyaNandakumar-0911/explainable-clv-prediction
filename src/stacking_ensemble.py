"""
Stage 8 - Deep-Ensemble: Multi-Model Stacking on Entity Embeddings
Deep-Ensemble CLV Research Project

This is the final proposed model: four structurally different base
learners (Random Forest, XGBoost, LightGBM, CatBoost) trained on the
entity-embedding feature set from Stage 4, combined via a Ridge
meta-learner trained on out-of-fold predictions (5-fold CV, so the
meta-learner never sees a base learner's prediction on data it was
trained on -- no leakage).

Why this counts as a genuine "deep ensemble": the base learners differ
in how they split, regularize, and handle interactions, so they make
different kinds of errors on different customers. The meta-learner
learns how much to trust each one, rather than averaging blindly.

Run from repo root:
    python src/stacking_ensemble.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42
TARGET = "true_clv"
N_FOLDS = 5

BASE_MODEL_FACTORY = {
    "rf": lambda: RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1),
    "xgb": lambda: XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE),
    "lgbm": lambda: LGBMRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
        random_state=RANDOM_STATE, verbosity=-1),
    "catboost": lambda: CatBoostRegressor(
        iterations=600, depth=5, learning_rate=0.03,
        random_state=RANDOM_STATE, verbose=False),
}


def main():
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(f"{PROCESSED_DIR}/embedded_features.csv")
    train_ids = pd.read_csv(f"{PROCESSED_DIR}/split_train_ids.csv")["customer_id"]
    test_ids = pd.read_csv(f"{PROCESSED_DIR}/split_test_ids.csv")["customer_id"]

    train_mask = df["customer_id"].isin(train_ids)
    test_mask = df["customer_id"].isin(test_ids)
    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]

    X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
    y_train = df.loc[train_mask, TARGET].reset_index(drop=True)
    X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)
    y_test = df.loc[test_mask, TARGET].reset_index(drop=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = pd.DataFrame(index=X_train.index, columns=BASE_MODEL_FACTORY.keys(), dtype=float)
    test_preds = pd.DataFrame(index=X_test.index, columns=BASE_MODEL_FACTORY.keys(), dtype=float)
    fitted_full_models = {}  # trained on ALL train data, for SHAP + deployment later

    print(f"Training {len(BASE_MODEL_FACTORY)} base learners with {N_FOLDS}-fold OOF stacking...\n")
    for name, make_model in BASE_MODEL_FACTORY.items():
        fold_test_preds = []
        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            model = make_model()
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            oof_preds.loc[val_idx, name] = model.predict(X_train.iloc[val_idx])
            fold_test_preds.append(model.predict(X_test))
        test_preds[name] = np.mean(fold_test_preds, axis=0)

        standalone_r2 = r2_score(y_test, test_preds[name])
        print(f"  {name:10s} standalone test R2: {standalone_r2:.4f}")

        # also fit on full train set for later use (SHAP, deployment)
        full_model = make_model()
        full_model.fit(X_train, y_train)
        fitted_full_models[name] = full_model

    # meta-learner: linear blend of out-of-fold predictions -> no leakage
    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds, y_train)
    final_test_pred = meta.predict(test_preds)

    mae = mean_absolute_error(y_test, final_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
    r2 = r2_score(y_test, final_test_pred)

    print(f"\n{'STACKED DEEP-ENSEMBLE (proposed)':35s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  R2={r2:.4f}")
    print(f"Meta-learner weights: {dict(zip(oof_preds.columns, meta.coef_))}")

    # append to comparison table
    results_df = pd.read_csv(f"{PROCESSED_DIR}/baseline_results.csv")
    new_row = pd.DataFrame([{
        "model": "Deep-Ensemble (RF+XGB+LGBM+CatBoost, embeddings, Ridge meta)",
        "mae": mae, "rmse": rmse, "r2": r2
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(f"{PROCESSED_DIR}/baseline_results.csv", index=False)

    print("\n--- Full comparison table ---")
    print(results_df.to_string(index=False))

    # save everything Stage 9 (SHAP) will need
    with open(f"{MODELS_DIR}/stacking_base_models.pkl", "wb") as f:
        pickle.dump(fitted_full_models, f)
    with open(f"{MODELS_DIR}/stacking_meta_model.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved base models to {MODELS_DIR}/stacking_base_models.pkl")
    print(f"Saved meta-learner to {MODELS_DIR}/stacking_meta_model.pkl")


if __name__ == "__main__":
    main()
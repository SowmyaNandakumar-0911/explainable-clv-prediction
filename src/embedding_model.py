"""
Stage 4 - Entity Embedding Training
Deep-Ensemble CLV Research Project

Trains a small feed-forward neural network with a learned embedding layer
per categorical column, jointly with the numeric features, to predict
true_clv. Once trained, we don't use this network's own predictions --
we extract its learned embedding vectors and use them as engineered
features for Stage 5's gradient boosting model.

This is the core idea behind entity embeddings (Guo & Berkhahn, 2016):
instead of one-hot encoding categories as independent columns, let the
network learn a dense vector per category so similar categories
(e.g. two segments that behave alike) end up close together in vector
space, capturing relationships one-hot encoding can't.

Run from repo root:
    python src/embedding_model.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def embedding_dim_for(cardinality: int) -> int:
    # common rule of thumb (Guo & Berkhahn / fast.ai): min(50, (cardinality+1)//2)
    # capped lower here since our cardinalities are small (3-12 categories)
    return min(8, max(2, (cardinality + 1) // 2))


class EntityEmbeddingNet(nn.Module):
    def __init__(self, cat_cardinalities: dict, n_numeric: int):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, cardinality in cat_cardinalities.items():
            dim = embedding_dim_for(cardinality)
            self.embeddings[col] = nn.Embedding(cardinality, dim)
            total_emb_dim += dim

        input_dim = total_emb_dim + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cat_inputs: dict, numeric_input):
        emb_outputs = [self.embeddings[col](cat_inputs[col]) for col in self.cat_cols]
        x = torch.cat(emb_outputs + [numeric_input], dim=1)
        return self.mlp(x).squeeze(-1)


def encode_categoricals(df: pd.DataFrame):
    """Label-encode each categorical column; return encoded df + category maps."""
    encoded = df.copy()
    cat_maps = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        encoded[col] = df[col].map(mapping)
        cat_maps[col] = mapping
    return encoded, cat_maps


def main():
    df = pd.read_csv(f"{PROCESSED_DIR}/model_table.csv")
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    train_ids = pd.read_csv(f"{PROCESSED_DIR}/split_train_ids.csv")["customer_id"]
    test_ids = pd.read_csv(f"{PROCESSED_DIR}/split_test_ids.csv")["customer_id"]

    df_encoded, cat_maps = encode_categoricals(df)
    df_encoded[NUMERIC_COLS] = df_encoded[NUMERIC_COLS].astype(float)

    scaler = StandardScaler()
    train_mask = df_encoded["customer_id"].isin(train_ids)
    df_encoded.loc[train_mask, NUMERIC_COLS] = scaler.fit_transform(
        df_encoded.loc[train_mask, NUMERIC_COLS])
    df_encoded.loc[~train_mask, NUMERIC_COLS] = scaler.transform(
        df_encoded.loc[~train_mask, NUMERIC_COLS])

    cat_cardinalities = {col: len(mapping) for col, mapping in cat_maps.items()}
    numeric_feature_cols = NUMERIC_COLS + BOOLEAN_COLS

    def to_tensors(mask):
        cat_inputs = {
            col: torch.tensor(df_encoded.loc[mask, col].values, dtype=torch.long)
            for col in CATEGORICAL_COLS
        }
        numeric_input = torch.tensor(
            df_encoded.loc[mask, numeric_feature_cols].values, dtype=torch.float32)
        target = torch.tensor(df_encoded.loc[mask, TARGET].values, dtype=torch.float32)
        return cat_inputs, numeric_input, target

    train_cat, train_num, train_y = to_tensors(train_mask)
    test_cat, test_num, test_y = to_tensors(~train_mask)

    model = EntityEmbeddingNet(cat_cardinalities, n_numeric=len(numeric_feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n_epochs = 60
    batch_size = 256
    n_train = train_num.shape[0]

    print("Training entity embedding network...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_cat = {col: train_cat[col][idx] for col in CATEGORICAL_COLS}
            batch_num = train_num[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(test_cat, test_num)
                test_loss = loss_fn(test_preds, test_y).item()
            print(f"Epoch {epoch:3d}  train_mse={epoch_loss / n_train:10.1f}  "
                  f"test_mse={test_loss:10.1f}")

    # extract learned embedding vectors per category, save for inspection
    # and for Stage 6's SHAP-to-category mapping
    embedding_lookup = {}
    for col in CATEGORICAL_COLS:
        weights = model.embeddings[col].weight.detach().numpy()
        embedding_lookup[col] = {cat: weights[idx].tolist()
                                  for cat, idx in cat_maps[col].items()}

    import json
    with open(f"{PROCESSED_DIR}/embedding_lookup.json", "w") as f:
        json.dump(embedding_lookup, f, indent=2)

    # build the full embedded feature table: numeric features + looked-up
    # embedding vectors for every row, for both train and test customers
    def build_embedded_row_features(mask):
        cat_inputs = {col: torch.tensor(df_encoded.loc[mask, col].values, dtype=torch.long)
                      for col in CATEGORICAL_COLS}
        with torch.no_grad():
            emb_parts = [model.embeddings[col](cat_inputs[col]).numpy()
                         for col in CATEGORICAL_COLS]
        emb_block = np.concatenate(emb_parts, axis=1)
        emb_cols = []
        for col in CATEGORICAL_COLS:
            dim = model.embeddings[col].embedding_dim
            emb_cols += [f"{col}_emb_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(emb_block, columns=emb_cols, index=df_encoded.loc[mask].index)
        numeric_df = df_encoded.loc[mask, numeric_feature_cols].reset_index(drop=True)
        emb_df = emb_df.reset_index(drop=True)
        ids = df_encoded.loc[mask, "customer_id"].reset_index(drop=True)
        target = df_encoded.loc[mask, TARGET].reset_index(drop=True)
        return pd.concat([ids, numeric_df, emb_df, target], axis=1)

    full_embedded = pd.concat([
        build_embedded_row_features(train_mask),
        build_embedded_row_features(~train_mask),
    ], axis=0).reset_index(drop=True)

    full_embedded.to_csv(f"{PROCESSED_DIR}/embedded_features.csv", index=False)
    print(f"\nSaved embedded feature table to {PROCESSED_DIR}/embedded_features.csv")
    print(f"Saved raw embedding vectors (per category) to {PROCESSED_DIR}/embedding_lookup.json")
    print(f"Embedding dims used: "
          f"{ {col: embedding_dim_for(c) for col, c in cat_cardinalities.items()} }")


if __name__ == "__main__":
    main()
    
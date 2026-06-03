import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from models.splitter import purged_walk_forward_splits
import config

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4.0,
    eval_metric="auc",
    random_state=42,
    n_jobs=2,
    tree_method="hist",
)


def train_symbol(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[XGBClassifier, dict]:
    df = feat_df.dropna(subset=feature_cols + ["label"]).copy()
    X = df[feature_cols].values
    y = df["label"].astype(int).values

    aucs = []
    for tr, va in purged_walk_forward_splits(
        df.index, n_splits=5, purge_gap=config.PURGE_GAP_DAYS,
    ):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X[tr], y[tr])
        pred = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], pred))

    final = XGBClassifier(**XGB_PARAMS)
    final.fit(X, y)

    importances = dict(zip(feature_cols, final.feature_importances_.tolist()))
    top10 = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:10])

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "cv_auc_mean": float(np.mean(aucs)) if aucs else None,
        "cv_auc_std": float(np.std(aucs)) if aucs else None,
        "cv_n_folds": len(aucs),
        "positive_rate": float(y.mean()),
        "top10_importance": top10,
    }
    return final, meta


def save_model(symbol: str, model: XGBClassifier, meta: dict):
    config.STORAGE_MODELS.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace(".", "_").replace("^", "")
    joblib.dump(model, config.STORAGE_MODELS / f"{safe}.pkl")
    (config.STORAGE_MODELS / f"{safe}_meta.json").write_text(
        json.dumps(meta, indent=2)
    )


def load_model(symbol: str) -> tuple[XGBClassifier, dict]:
    safe = symbol.replace(".", "_").replace("^", "")
    model = joblib.load(config.STORAGE_MODELS / f"{safe}.pkl")
    meta = json.loads((config.STORAGE_MODELS / f"{safe}_meta.json").read_text())
    return model, meta

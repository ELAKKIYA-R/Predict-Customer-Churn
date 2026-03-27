#!/usr/bin/env python3
"""
Stratified K-fold training for Playground Series S6E3 (Telco churn).
Supports xgboost, lightgbm, catboost. Writes averaged test predictions to submission CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parent

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUM_COLS_BASE = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tenure_safe = out["tenure"].replace(0, np.nan)
    out["avg_monthly_total"] = out["TotalCharges"] / tenure_safe
    out["avg_monthly_total"] = out["avg_monthly_total"].fillna(out["MonthlyCharges"])
    out["monthly_x_tenure"] = out["MonthlyCharges"] * out["tenure"]
    return out


def target_to_binary(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).to_numpy()
    m = {"Yes": 1, "No": 0}
    return s.map(m).astype(int).to_numpy()


def build_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("num", "passthrough", num_cols), ("cat", ohe, CAT_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
) -> tuple[object, float]:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=80,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    p_va = model.predict_proba(X_va)[:, 1]
    return model, roc_auc_score(y_va, p_va)


def train_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
) -> tuple[object, float]:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=0)],
    )
    p_va = model.predict_proba(X_va)[:, 1]
    return model, roc_auc_score(y_va, p_va)


def train_catboost(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    df_va: pd.DataFrame,
    y_va: np.ndarray,
    num_cols: list[str],
    seed: int,
) -> tuple[object, float]:
    from catboost import CatBoostClassifier, Pool

    cat_idx = [df_tr.columns.get_loc(c) for c in CAT_COLS]
    train_pool = Pool(df_tr, label=y_tr, cat_features=cat_idx)
    valid_pool = Pool(df_va, label=y_va, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        early_stopping_rounds=80,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, verbose=False)
    p_va = model.predict_proba(df_va)[:, 1]
    return model, roc_auc_score(y_va, p_va)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder containing train.csv, test.csv",
    )
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "submission.csv",
        help="Output submission path",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path to write fold AUCs as JSON",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Need train.csv and test.csv under {data_dir}")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    y = target_to_binary(train_raw["Churn"])
    train_df = add_features(train_raw.drop(columns=["Churn"]))
    test_df = add_features(test_raw)

    num_cols = NUM_COLS_BASE + ["avg_monthly_total", "monthly_x_tenure"]
    feature_cols = ["id"] + num_cols + CAT_COLS
    for c in feature_cols:
        if c not in train_df.columns:
            raise KeyError(f"Missing column {c}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    test_pred_sum = np.zeros(len(test_df), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        if args.model == "catboost":
            cols = num_cols + CAT_COLS
            model, auc_va = train_catboost(
                tr[cols], y_tr, va[cols], y_va, num_cols, args.seed + fold
            )
            p_test = model.predict_proba(test_df[cols])[:, 1]
        else:
            pre = build_preprocessor(num_cols)
            X_tr = pre.fit_transform(tr)
            X_va = pre.transform(va)
            X_test = pre.transform(test_df)

            if args.model == "xgboost":
                model, auc_va = train_xgb(X_tr, y_tr, X_va, y_va, args.seed + fold)
            else:
                model, auc_va = train_lgbm(X_tr, y_tr, X_va, y_va, args.seed + fold)
            p_test = model.predict_proba(X_test)[:, 1]

        fold_aucs.append(float(auc_va))
        test_pred_sum += p_test
        print(f"Fold {fold + 1}/{args.n_folds}  val AUC = {auc_va:.5f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")

    test_pred = test_pred_sum / args.n_folds
    sub = pd.DataFrame({"id": test_df["id"], "Churn": test_pred})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} (mean probability over folds)")

    if args.metrics_json:
        payload = {
            "model": args.model,
            "n_folds": args.n_folds,
            "fold_aucs": fold_aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        }
        args.metrics_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()

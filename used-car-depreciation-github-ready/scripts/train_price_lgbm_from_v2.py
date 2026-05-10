#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Train a LightGBM regressor on a cleaned used-car dataset.

The script uses a time-based validation split when `listed_date` is available,
falls back to a random split otherwise, and saves model artifacts for later
scoring and demo use.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from sklearn.metrics import root_mean_squared_error as rmse
except Exception:
    from sklearn.metrics import mean_squared_error

    def rmse(y, p):
        return mean_squared_error(y, p, squared=False)


def rmape(y, p, eps=1.0):
    y = np.asarray(y)
    p = np.asarray(p)
    return np.mean(np.abs(p - y) / np.maximum(np.abs(y), eps))


def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[,$ ]", "", regex=True).replace({"": np.nan}),
        errors="coerce",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "cars_clean_v2.csv"))
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "artifacts"))
    ap.add_argument("--valid-quantile", type=float, default=0.80, help="Listed-date quantile used for validation split.")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--row-cap", type=int, default=0, help="Optionally sample N rows for a quick smoke test.")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    for col in [
        "price",
        "mileage",
        "horsepower",
        "engine_displacement",
        "engine_disp_l",
        "owner_count",
        "daysonmarket",
        "year",
        "cylinders_num",
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])
    if "listed_date" in df.columns:
        df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")

    for col in ["fleet", "frame_damaged", "has_accidents"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            true_set = {"t", "true", "1", "yes", "y"}
            false_set = {"f", "false", "0", "no", "n"}
            mapped = s.map(lambda x: True if x in true_set else False if x in false_set else np.nan)
            df[col] = mapped.astype("Int8").fillna(0)

    if "engine_disp_l" not in df.columns and "engine_displacement" in df.columns:
        disp = df["engine_displacement"]
        df["engine_disp_l"] = disp / 1000.0 if disp.dropna().gt(10).mean() > 0.5 else disp

    if "mileage" in df.columns:
        df["log_mileage"] = np.log10(np.clip(df["mileage"], 1, None))
    if {"horsepower", "engine_disp_l"}.issubset(df.columns):
        df["hp_per_l"] = df["horsepower"] / df["engine_disp_l"].replace(0, np.nan)
    if "age_years" not in df.columns and {"listed_date", "year"}.issubset(df.columns):
        df["age_years"] = df["listed_date"].dt.year - df["year"]
    if "age_years" in df.columns:
        df["age_years"] = df["age_years"].clip(lower=0)
    df["age_sq"] = df.get("age_years", pd.Series(index=df.index, dtype=float)) ** 2

    if args.row_cap and len(df) > args.row_cap:
        df = df.sample(n=args.row_cap, random_state=args.seed)
        print(f"[info] sampled rows: {len(df):,}")

    if "price" not in df.columns:
        raise ValueError("Column 'price' not found in dataset.")
    df["y"] = np.log1p(df["price"])

    num_cols = [
        "mileage",
        "log_mileage",
        "age_years",
        "age_sq",
        "horsepower",
        "engine_disp_l",
        "cylinders_num",
        "owner_count",
        "hp_per_l",
        "has_accidents",
        "frame_damaged",
        "fleet",
    ]
    cat_cols = [
        "make_name",
        "model_name",
        "body_type",
        "fuel_type",
        "transmission",
        "wheel_system",
        "trim_name",
        "engine_type_std",
        "interior_color",
        "listing_color",
    ]

    if "daysonmarket" in df.columns:
        df.drop(columns=["daysonmarket"], inplace=True)

    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    feat_cols = [c for c in (num_cols + cat_cols) if c in df.columns]
    if not feat_cols:
        raise ValueError("No feature columns available after intersection. Check dataset columns.")

    if "listed_date" in df.columns and df["listed_date"].notna().any():
        cut = df["listed_date"].dropna().sort_values().quantile(args.valid_quantile)
        train = df[df["listed_date"] <= cut].copy()
        valid = df[df["listed_date"] > cut].copy()
        split_note = f"time-split cut={str(cut.date())}  train={len(train):,}  valid={len(valid):,}"
    else:
        rng = np.random.RandomState(args.seed)
        mask = rng.rand(len(df)) < 0.8
        train, valid = df[mask].copy(), df[~mask].copy()
        cut = None
        split_note = f"random-split seed={args.seed}  train={len(train):,}  valid={len(valid):,}"
    print("[split]", split_note)

    X_tr, y_tr = train[feat_cols], train["y"]
    X_va, y_va = valid[feat_cols], valid["y"]

    lgb_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    lgb_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=lgb_tr, free_raw_data=False)

    params = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=args.learning_rate,
        num_leaves=128,
        min_data_in_leaf=400,
        feature_fraction=0.75,
        bagging_fraction=0.75,
        bagging_freq=1,
        max_bin=255,
        lambda_l2=5.0,
        verbosity=1,
        seed=args.seed,
        num_threads=os.cpu_count(),
    )

    def progress_callback(period=50):
        def _cb(env):
            if env.iteration == 1 or env.iteration % period == 0:
                parts = []
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    if data_name == "valid" and eval_name in ("rmse", "l2"):
                        parts.append(f"{data_name}'s {eval_name}: {result:.6f}")
                if parts:
                    print(f"[{env.iteration}] " + "  ".join(parts))
                    sys.stdout.flush()

        _cb.order = 10
        return _cb

    print("[train] starting LightGBM ...")
    sys.stdout.flush()

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping),
        progress_callback(period=50),
    ]
    if hasattr(lgb, "log_evaluation"):
        callbacks.insert(0, lgb.log_evaluation(period=50))

    model = lgb.train(
        params,
        lgb_tr,
        num_boost_round=args.iterations,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    pred_log = model.predict(X_va, num_iteration=getattr(model, "best_iteration", None))
    pred = np.expm1(pred_log)
    y_true = np.expm1(y_va)

    metrics = {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(mean_squared_error(y_true, pred, squared=False)),
        "R2": float(r2_score(y_true, pred)),
        "rMAPE": float(np.mean(np.abs(pred - y_true) / np.maximum(np.abs(y_true), 1.0))),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "features": feat_cols,
        "categoricals": cat_cols,
        "split": split_note,
        "params": params,
    }
    print("\n[valid metrics]", {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})

    model_path = out_dir / "lgbm_price_model.joblib"
    joblib.dump(
        dict(model=model, features=feat_cols, cats=cat_cols, cut=str(cut) if cut is not None else None, params=params),
        model_path,
    )

    imp = pd.DataFrame(
        {
            "feature": feat_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    keep_cols = [
        c
        for c in [
            "price",
            "mileage",
            "age_years",
            "horsepower",
            "engine_disp_l",
            "make_name",
            "model_name",
            "body_type",
            "fuel_type",
            "transmission",
            "wheel_system",
            "trim_name",
        ]
        if c in valid.columns
    ]
    pred_df = pd.DataFrame(
        {
            "price": valid["price"].values,
            "pred": pred,
            "residual": pred - valid["price"].values,
        }
    )
    for c in keep_cols:
        pred_df[c] = valid[c].values
    pred_df.to_csv(out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "training_log.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\n[saved] model -> {model_path}")
    print(f"[saved] feature_importance.csv, valid_predictions.csv, training_log.txt in {out_dir}")


if __name__ == "__main__":
    main()

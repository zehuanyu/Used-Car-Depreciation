#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a LightGBM regressor on the FULL dataset (no train/valid/test split).
Optionally inherit params/features/categorical cols & best rounds from a previous bundle.

Usage (PowerShell):
  python final_train_on_all.py `
    --data "C:\\Users\\92963\\Desktop\\research\\cars_clean_v2.csv" `
    --out  "C:\\Users\\92963\\Desktop\\research\\artifacts_full2" `
    --base-bundle "C:\\Users\\92963\\Desktop\\research\\artifacts_full\\lgbm_price_model.joblib" `
    --iterations 1642
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------- metrics -------
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except Exception:
    from sklearn.metrics import mean_squared_error
    def rmse(y, p): return mean_squared_error(y, p, squared=False)

def rmape(y, p, eps=1.0):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), eps)))

# ------- helpers -------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[,$ ]", "", regex=True).replace({"": np.nan}),
        errors="coerce"
    )

def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make preprocessing consistent with your previous training."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # numeric coercions
    for col in ["price","mileage","horsepower","engine_displacement",
                "engine_disp_l","owner_count","daysonmarket","year","cylinders_num"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # dates
    if "listed_date" in df.columns:
        df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")

    # booleans -> Int8(0/1)
    for col in ["fleet","frame_damaged","has_accidents"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            true_set, false_set = {"t","true","1","yes","y"}, {"f","false","0","no","n"}
            mapped = s.map(lambda x: True if x in true_set else False if x in false_set else np.nan)
            df[col] = mapped.astype("Int8").fillna(0)

    # engine_disp_l derive if missing
    if "engine_disp_l" not in df.columns and "engine_displacement" in df.columns:
        disp = df["engine_displacement"]
        if disp.dropna().gt(10).mean() > 0.5:  # looks like cc
            df["engine_disp_l"] = disp / 1000.0
        else:
            df["engine_disp_l"] = disp

    # derived
    if "mileage" in df.columns:
        df["log_mileage"] = np.log10(np.clip(df["mileage"], 1, None))
    if {"horsepower","engine_disp_l"}.issubset(df.columns):
        df["hp_per_l"] = df["horsepower"] / df["engine_disp_l"].replace(0, np.nan)
    if "age_years" not in df.columns and {"listed_date","year"}.issubset(df.columns):
        list_year = df["listed_date"].dt.year
        df["age_years"] = list_year - df["year"]
    if "age_years" in df.columns:
        df["age_years"] = df["age_years"].clip(lower=0)
    df["age_sq"] = df.get("age_years", pd.Series(index=df.index, dtype=float)) ** 2

    # optional: drop known leaky/unstable columns
    if "daysonmarket" in df.columns:
        df.drop(columns=["daysonmarket"], inplace=True)

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "cars_clean_v2.csv"))
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "artifacts_full"))
    ap.add_argument("--base-bundle", type=str, default="",
                    help="previous joblib bundle to inherit params/features/cats/best_iteration")
    ap.add_argument("--iterations", type=int, default=0,
                    help="num_boost_round for full training; if 0, try base bundle best_iteration; else use default 1200")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="override learning_rate (if provided)")
    ap.add_argument("--row-cap", type=int, default=0, help="sample first N rows for quick run")
    args = ap.parse_args()

    DATA = Path(args.data)
    OUT  = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading: {DATA}")
    raw = pd.read_csv(DATA, low_memory=False)
    if args.row_cap and len(raw) > args.row_cap:
        raw = raw.iloc[:args.row_cap].copy()
        print(f"[info] sampled rows: {len(raw):,}")

    df = prep_features(raw)

    # ---- target ----
    if "price" not in df.columns:
        raise ValueError("Column 'price' not found in dataset.")
    y_log = np.log1p(df["price"].astype(float))

    # ---- load base bundle or set defaults ----
    feat_cols, cat_cols, params, best_rounds = None, [], None, None
    source_bundle = None

    if args.base_bundle:
        source_bundle = Path(args.base_bundle)
        print(f"[info] inherit from: {source_bundle}")
        bundle = joblib.load(source_bundle)
        feat_cols = bundle.get("features", None)
        cat_cols  = bundle.get("cats", [])
        params    = bundle.get("params", None)
        # prefer best_iteration from training history if available
        best_rounds = bundle.get("best_iteration", None)

    # default feature candidates if no bundle
    if feat_cols is None:
        num_cols = [
            "mileage","log_mileage","age_years","age_sq",
            "horsepower","engine_disp_l","cylinders_num","owner_count","hp_per_l",
            "has_accidents","frame_damaged","fleet",
        ]
        cat_cols = [
            "make_name","model_name","body_type","fuel_type","transmission",
            "wheel_system","trim_name","engine_type_std"
        ]
        feat_cols = [c for c in num_cols + cat_cols if c in df.columns]
        print(f"[warn] no base bundle provided; using default feature intersection ({len(feat_cols)} cols).")

    # ensure categoricals
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    X = df[feat_cols].copy()

    # ---- params ----
    if params is None:
        params = dict(
            objective="regression", metric="rmse", boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=128,
            min_data_in_leaf=400,
            feature_fraction=0.75,
            bagging_fraction=0.75, bagging_freq=1,
            max_bin=255,
            lambda_l2=5.0,
            seed=42,
            num_threads=os.cpu_count(),
            force_col_wise=True,
            min_data_in_bin=3,
        )
        print("[warn] using default params.")
    if args.learning_rate is not None:
        params["learning_rate"] = float(args.learning_rate)

    # determine rounds
    if args.iterations > 0:
        final_rounds = int(args.iterations)
    elif isinstance(best_rounds, (int, float)) and best_rounds > 0:
        final_rounds = int(best_rounds)
    else:
        final_rounds = 1200  # safe default
    print(f"[info] num_boost_round = {final_rounds}")

    # ---- train on ALL data (no early stopping, no valid sets) ----
    lgb_all = lgb.Dataset(X, label=y_log, categorical_feature=[c for c in cat_cols if c in X.columns], free_raw_data=False)
    print("[train] starting LightGBM (full data) ...")
    model = lgb.train(params, lgb_all, num_boost_round=final_rounds)

    # ---- in-sample evaluation (for sanity check only) ----
    pred_log = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    pred = np.expm1(pred_log)
    y_true = df["price"].values

    metrics = {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(rmse(y_true, pred)),
        "R2": float(r2_score(y_true, pred)),
        "rMAPE": float(rmape(y_true, pred)),
        "n": int(len(df)),
        "features": feat_cols,
        "categoricals": cat_cols,
        "params": params,
        "num_boost_round": int(final_rounds),
        "source_bundle": str(source_bundle) if source_bundle else None
    }
    print("\n[in-sample metrics] ", {k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in metrics.items()})

    # ---- save artifacts ----
    out_model = OUT / "lgbm_price_model_full.joblib"
    joblib.dump(
        dict(model=model, features=feat_cols, cats=cat_cols, params=params, cut=None,
             best_iteration=int(getattr(model, "best_iteration", final_rounds))),
        out_model
    )

    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # in-sample predictions (for audit only)
    keep_cols = [c for c in ["price","mileage","age_years","horsepower","engine_disp_l",
                             "make_name","model_name","body_type","fuel_type",
                             "transmission","wheel_system","trim_name"] if c in df.columns]
    pred_df = pd.DataFrame({"pred": pred, "price": y_true})
    pred_df["residual"] = pred_df["pred"] - pred_df["price"]
    for c in keep_cols:
        pred_df[c] = df[c].values
    pred_df.to_csv(OUT / "in_sample_predictions.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "training_log.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\n[saved] model -> {out_model}")
    print(f"[saved] feature_importance.csv, in_sample_predictions.csv, training_log.json in {OUT}")

if __name__ == "__main__":
    main()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

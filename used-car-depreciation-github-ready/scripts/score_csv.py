#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Score a vehicle dataset with a trained LightGBM price model bundle.

If the dataset includes `price`, the script also computes MAE, RMSE, R2, and
relative MAPE. It can optionally create train/validation/test subsets based on
the model's saved cutoff date, explicit dates, or listed-date quantiles.

Examples:
  python score_csv.py `
    --model "models\lgbm_price_model.joblib" `
    --data "data\processed\cars_clean_v2.csv" `
    --out "results\scored" `
    --segments "make_name,body_type"

  python score_csv.py --use-model-cut
  python score_csv.py --cut-date 2020-08-29 --test-cut-date 2021-01-01
  python score_csv.py --split-quantiles 0.80,0.90
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

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
    return float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), eps)))


def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[,$ ]", "", regex=True).replace({"": np.nan}),
        errors="coerce",
    )


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df


def compute_metrics(df_sub: pd.DataFrame) -> dict:
    if "price" not in df_sub.columns or df_sub["price"].isna().all():
        return {}
    y_true = df_sub["price"].values
    y_pred = df_sub["pred_price"].values
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "rMAPE": float(rmape(y_true, y_pred)),
        "n": int(len(df_sub)),
    }


def save_segment_report(df_sub: pd.DataFrame, seg_cols: list[str], path: Path):
    if not seg_cols or any(c not in df_sub.columns for c in seg_cols):
        return

    grp = df_sub.groupby(seg_cols, dropna=False, observed=True)

    def residual_rmape(r, base):
        return float(np.mean(np.abs(r.values) / np.maximum(base.loc[r.index, "price"].values, 1.0)))

    rep = (
        grp.agg(
            n=("pred_price", "size"),
            MAE=("residual", lambda r: float(np.mean(np.abs(r)))),
            RMSE=("residual", lambda r: float(np.sqrt(np.mean(np.square(r))))),
            rMAPE=("residual", lambda r, base=df_sub: residual_rmape(r, base)),
            bias=("residual", "mean"),
        )
        .reset_index()
        .sort_values("RMSE", ascending=False)
    )
    rep.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "lgbm_price_model.joblib"))
    ap.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "cars_clean_v2.csv"))
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "scored"))
    ap.add_argument("--segments", type=str, default="", help="Comma-separated columns for grouped metrics.")
    ap.add_argument("--row-cap", type=int, default=0, help="Optionally score only the first N rows.")
    ap.add_argument("--query", type=str, default="", help="Optional pandas query filter.")
    ap.add_argument("--keep-cols", type=str, default="", help="Extra columns to preserve in outputs.")
    ap.add_argument("--use-model-cut", action="store_true", help="Reuse the cutoff date saved in the model bundle.")
    ap.add_argument("--cut-date", type=str, default="", help="Train/validation cutoff date in YYYY-MM-DD format.")
    ap.add_argument("--test-cut-date", type=str, default="", help="Validation/test cutoff date in YYYY-MM-DD format.")
    ap.add_argument("--split-quantiles", type=str, default="", help="Two listed-date quantiles such as 0.80,0.90.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(Path(args.model))
    model = bundle["model"]
    feat_cols = bundle["features"]
    cat_cols = bundle.get("cats", [])
    model_cut = bundle.get("cut", None)

    print(f"[info] reading: {args.data}")
    raw = pd.read_csv(args.data, low_memory=False)
    if args.row_cap and len(raw) > args.row_cap:
        raw = raw.iloc[: args.row_cap].copy()
        print(f"[info] sampled rows: {len(raw):,}")

    df = prep_features(raw)
    if args.query:
        before = len(df)
        df = df.query(args.query).copy()
        print(f"[filter] query='{args.query}' -> {before:,} -> {len(df):,}")

    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c] if c in df.columns else np.nan
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    pred_log = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    pred = np.expm1(pred_log)

    out_df = df.copy()
    out_df["pred_price"] = pred
    if "price" in out_df.columns:
        out_df["residual"] = out_df["pred_price"] - out_df["price"]

    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    for c in keep_cols:
        if c in df.columns:
            out_df[c] = df[c].values

    out_df.to_csv(out_dir / "scored_all.csv", index=False, encoding="utf-8-sig")
    out_df.to_csv(out_dir / "scored.csv", index=False, encoding="utf-8-sig")
    print(f"[saved] {out_dir / 'scored_all.csv'} (and scored.csv)")

    if "price" in out_df.columns:
        metrics_all = compute_metrics(out_df)
        if metrics_all:
            (out_dir / "metrics_all.json").write_text(json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / "metrics.json").write_text(json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding="utf-8")
            print("[metrics_all]", metrics_all)

        seg_cols = [s.strip() for s in args.segments.split(",") if s.strip()]
        if seg_cols:
            save_segment_report(out_df, seg_cols, out_dir / "segment_report_all.csv")
            (out_dir / "segment_report.csv").write_text(
                (out_dir / "segment_report_all.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"[saved] {out_dir / 'segment_report_all.csv'} (and segment_report.csv)")

    cut_date = None
    test_cut_date = None

    if args.use_model_cut and model_cut:
        try:
            cut_date = pd.to_datetime(model_cut).normalize()
        except Exception:
            pass

    if args.cut_date:
        cut_date = pd.to_datetime(args.cut_date).normalize()
    if args.test_cut_date:
        test_cut_date = pd.to_datetime(args.test_cut_date).normalize()

    if args.split_quantiles and ("listed_date" in out_df.columns) and out_df["listed_date"].notna().any():
        parts = [x.strip() for x in args.split_quantiles.split(",") if x.strip()]
        if len(parts) >= 1 and cut_date is None:
            cut_date = out_df["listed_date"].dropna().quantile(float(parts[0])).normalize()
        if len(parts) >= 2 and test_cut_date is None:
            test_cut_date = out_df["listed_date"].dropna().quantile(float(parts[1])).normalize()

    if ("listed_date" in out_df.columns) and out_df["listed_date"].notna().any() and cut_date is not None:
        ld = out_df["listed_date"]
        train_mask = (ld.notna()) & (ld <= cut_date)
        if test_cut_date is not None:
            valid_mask = (ld.notna()) & (ld > cut_date) & (ld <= test_cut_date)
            test_mask = (ld.notna()) & (ld > test_cut_date)
        else:
            valid_mask = (ld.notna()) & (ld > cut_date)
            test_mask = None

        def save_split(name: str, mask):
            sub = out_df.loc[mask].copy()
            sub.to_csv(out_dir / f"scored_{name}.csv", index=False, encoding="utf-8-sig")
            print(f"[saved] {out_dir / f'scored_{name}.csv'}  n={len(sub):,}")
            if "price" in sub.columns and len(sub) > 0:
                metrics = compute_metrics(sub)
                (out_dir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[metrics_{name}]", metrics)
                seg_cols = [s.strip() for s in args.segments.split(",") if s.strip()]
                if seg_cols:
                    save_segment_report(sub, seg_cols, out_dir / f"segment_report_{name}.csv")
                    print(f"[saved] {out_dir / f'segment_report_{name}.csv'}")

        save_split("train", train_mask)
        save_split("valid", valid_mask)
        if test_mask is not None:
            save_split("test", test_mask)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Generate a depreciation curve using the trained LightGBM model bundle
(saved by train_price_lgbm_from_v2.py). Supports sweeping by mileage or age.

Examples (PowerShell):
  # sweep by mileage, fix age=2 years
  python predict_curve.py ^
    --model "C:\Users\me\artifacts_full\lgbm_price_model.joblib" ^
    --out   "C:\Users\me\curves" ^
    --sweep mileage --age 2.0 ^
    --make Toyota --model-name Camry --body-type Sedan --fuel-type Gasoline ^
    --transmission A --wheel-system FWD --trim-name "<OTHER>" --engine-type-std I4 ^
    --engine-disp-l 2.5 --cylinders-num 4 --has-accidents 0 --frame-damaged 0 --fleet 0

  # sweep by age, fix mileage=30000
  python predict_curve.py ^
    --model "C:\Users\me\artifacts_full\lgbm_price_model.joblib" ^
    --out   "C:\Users\me\curves" ^
    --sweep age --mileage 30000 ^
    --make Toyota --model-name Camry --body-type Sedan --fuel-type Gasoline ^
    --transmission A --wheel-system FWD --trim-name "<OTHER>" --engine-type-std I4 ^
    --engine-disp-l 2.5 --cylinders-num 4
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_base_row(args: argparse.Namespace) -> dict:
    """Create a base vehicle configuration dict from CLI args."""
    base = dict(
        # numeric (some may be unused if model doesn't require them)
        mileage=float(args.mileage),
        age_years=float(args.age),
        age_sq=float(args.age) ** 2,
        engine_disp_l=float(args.engine_disp_l) if args.engine_disp_l is not None else np.nan,
        cylinders_num=float(args.cylinders_num) if args.cylinders_num is not None else np.nan,
        has_accidents=int(args.has_accidents),
        frame_damaged=int(args.frame_damaged),
        fleet=int(args.fleet),

        # categoricals
        make_name=args.make,
        model_name=args.model_name,
        body_type=args.body_type,
        fuel_type=args.fuel_type,
        transmission=args.transmission,
        wheel_system=args.wheel_system,
        trim_name=args.trim_name,
        engine_type_std=args.engine_type_std,
    )
    # derived (only if mileage exists)
    base["log_mileage"] = float(np.log10(max(1.0, base["mileage"])))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "lgbm_price_model.joblib"), help="path to lgbm_price_model.joblib")
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "curves"))
    ap.add_argument("--sweep", choices=["mileage", "age"], default="mileage")

    # vehicle config (categoricals)
    ap.add_argument("--make", type=str, default="Toyota")
    ap.add_argument("--model-name", type=str, default="Camry")
    ap.add_argument("--body-type", type=str, default="Sedan")
    ap.add_argument("--fuel-type", type=str, default="Gasoline")
    ap.add_argument("--transmission", type=str, default="A")
    ap.add_argument("--wheel-system", type=str, default="FWD")
    ap.add_argument("--trim-name", type=str, default="<OTHER>")
    ap.add_argument("--engine-type-std", type=str, default="I4")

    # vehicle config (numeric)
    ap.add_argument("--engine-disp-l", type=float, default=2.5)
    ap.add_argument("--cylinders-num", type=float, default=4)
    ap.add_argument("--has-accidents", type=int, default=0)
    ap.add_argument("--frame-damaged", type=int, default=0)
    ap.add_argument("--fleet", type=int, default=0)

    # sweep controls
    ap.add_argument("--points", type=int, default=50)
    ap.add_argument("--mileage-min", type=float, default=1_000.0)
    ap.add_argument("--mileage-max", type=float, default=200_000.0)
    ap.add_argument("--age-min", type=float, default=0.0)
    ap.add_argument("--age-max", type=float, default=15.0)

    # anchors
    ap.add_argument("--mileage", type=float, default=30_000.0, help="fixed mileage when sweeping age")
    ap.add_argument("--age", type=float, default=2.0, help="fixed age when sweeping mileage")

    args = ap.parse_args()

    # Load model bundle
    bundle = joblib.load(Path(args.model))
    model = bundle["model"]
    feat_cols = bundle["features"]
    cat_cols = bundle.get("cats", [])

    # Prepare sweep values
    if args.sweep == "mileage":
        xs = np.logspace(np.log10(args.mileage_min), np.log10(args.mileage_max), args.points)
        x_label = "Mileage"
    else:
        xs = np.linspace(args.age_min, args.age_max, args.points)
        x_label = "Age (years)"

    # Build rows
    rows = []
    base = build_base_row(args)
    for x in xs:
        r = dict(base)  # copy
        if args.sweep == "mileage":
            r["mileage"] = float(x)
            r["log_mileage"] = float(np.log10(max(1.0, x)))
            # age fixed from args.age; age_sq already present in base -> ensure consistency:
            r["age_years"] = float(args.age)
            r["age_sq"] = float(args.age ** 2)
        else:
            r["age_years"] = float(x)
            r["age_sq"] = float(x ** 2)
            # mileage fixed from args.mileage
            r["mileage"] = float(args.mileage)
            r["log_mileage"] = float(np.log10(max(1.0, args.mileage)))

        rows.append(r)

    # Assemble to DataFrame and align features
    X = pd.DataFrame(rows)

    # Ensure all required features are present; missing ones as NaN (LightGBM can handle)
    for c in feat_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feat_cols]

    # Cast categoricals
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # Predict (model trained on log1p(price))
    pred_log = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    pred = np.expm1(pred_log)

    # Save outputs
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    base_tag = f"{args.make}_{args.model_name}_{args.trim_name}_{args.engine_type_std}".replace(" ", "_")
    csv_path = OUT / f"curve_{args.sweep}_{base_tag}.csv"
    png_path = OUT / f"curve_{args.sweep}_{base_tag}.png"

    pd.DataFrame({x_label: xs, "pred_price": pred}).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, pred, marker="o", markersize=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(f"Predicted price vs {x_label}")

    if args.sweep == "mileage":
        ax.set_xscale("log")

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"[saved] {csv_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from used_car_project.analysis import add_analysis_segments, load_dataset, prep_features, summarize_segment_errors


def main():
    ap = argparse.ArgumentParser(description="Generate segment-level error analysis from scored predictions.")
    ap.add_argument("--scored", type=str, required=True, help="Path to a scored CSV containing price, pred_price, and residual.")
    ap.add_argument("--out", type=str, default="results/error_analysis", help="Output directory.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prep_features(load_dataset(args.scored))
    df = add_analysis_segments(df)

    group_cols = ["make_name", "body_type", "price_band", "age_band", "mileage_band"]
    for col in group_cols:
        report = summarize_segment_errors(df, col)
        if not report.empty:
            report.to_csv(out_dir / f"{col}_error_report.csv", index=False, encoding="utf-8-sig")

    worst_rows = df.sort_values("residual", key=lambda s: s.abs(), ascending=False).head(100)
    keep_cols = [c for c in ["price", "pred_price", "residual", "make_name", "model_name", "body_type", "age_years", "mileage", "price_band", "age_band", "mileage_band"] if c in worst_rows.columns]
    worst_rows[keep_cols].to_csv(out_dir / "largest_prediction_errors.csv", index=False, encoding="utf-8-sig")
    print(f"[done] saved reports to {out_dir}")


if __name__ == "__main__":
    main()

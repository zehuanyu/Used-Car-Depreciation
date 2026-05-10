import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from used_car_project.analysis import load_dataset, prep_features, regression_metrics, select_model_features, split_time_or_random


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )


def evaluate_model(name: str, estimator, X_train, X_valid, y_train, y_valid) -> dict:
    estimator.fit(X_train, np.log1p(y_train))
    pred = np.expm1(estimator.predict(X_valid))
    metrics = regression_metrics(y_valid.to_numpy(), pred)
    metrics["model"] = name
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Benchmark multiple baseline models for used-car price prediction.")
    ap.add_argument("--data", type=str, required=True, help="CSV or parquet dataset path.")
    ap.add_argument("--out", type=str, default="results/benchmarks", help="Output directory.")
    ap.add_argument("--row-cap", type=int, default=30000, help="Maximum rows to use for benchmarking.")
    ap.add_argument("--valid-quantile", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1, help="Worker count for models that support parallel execution.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prep_features(load_dataset(args.data))
    df = df[df["price"].notna()].copy()
    if args.row_cap and len(df) > args.row_cap:
        df = df.sample(n=args.row_cap, random_state=args.seed).sort_index()

    num_cols, cat_cols = select_model_features(df)
    feature_cols = num_cols + cat_cols
    train, valid, split_note = split_time_or_random(df, valid_quantile=args.valid_quantile, seed=args.seed)

    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    y_train = train["price"]
    y_valid = valid["price"]

    preprocessor = make_preprocessor(num_cols, cat_cols)
    dense_preprocessor = make_preprocessor(num_cols, cat_cols)

    models = [
        ("Median Baseline", DummyRegressor(strategy="median")),
        ("Linear Regression", Pipeline([("prep", preprocessor), ("model", LinearRegression())])),
        (
            "Random Forest",
            Pipeline(
                [
                    ("prep", dense_preprocessor),
                    ("model", RandomForestRegressor(n_estimators=150, max_depth=18, random_state=args.seed, n_jobs=args.n_jobs)),
                ]
            ),
        ),
        (
            "Gradient Boosting",
            Pipeline(
                [
                    ("prep", make_preprocessor(num_cols, cat_cols)),
                    ("to_dense", FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x, validate=False)),
                    ("model", GradientBoostingRegressor(max_depth=4, learning_rate=0.06, random_state=args.seed)),
                ]
            ),
        ),
    ]

    rows = [evaluate_model(name, est, X_train, X_valid, y_train, y_valid) for name, est in models]
    result_df = pd.DataFrame(rows).sort_values("RMSE")
    result_df.to_csv(out_dir / "benchmark_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "split": split_note,
        "rows_used": int(len(df)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "best_model": result_df.iloc[0]["model"],
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result_df.to_string(index=False))
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()

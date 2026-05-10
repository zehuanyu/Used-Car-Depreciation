from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error as rmse
except Exception:
    from sklearn.metrics import mean_squared_error

    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "make_name",
    "model_name",
    "body_type",
    "fuel_type",
    "transmission",
    "wheel_system",
    "trim_name",
    "engine_type_std",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,$ ]", "", regex=True).replace({"": np.nan}),
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
        df["engine_disp_l"] = np.where(disp.dropna().gt(10).mean() > 0.5, disp / 1000.0, disp)

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


def select_model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_cols, cat_cols


def split_time_or_random(df: pd.DataFrame, valid_quantile: float = 0.8, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if "listed_date" in df.columns and df["listed_date"].notna().any():
        cut = df["listed_date"].dropna().sort_values().quantile(valid_quantile)
        train = df[df["listed_date"] <= cut].copy()
        valid = df[df["listed_date"] > cut].copy()
        note = f"time split at {cut.date()}"
        return train, valid, note

    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < valid_quantile
    train = df[mask].copy()
    valid = df[~mask].copy()
    return train, valid, f"random split seed={seed}"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "rMAPE": float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1.0))),
    }


def add_analysis_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price" in df.columns:
        df["price_band"] = pd.cut(
            df["price"],
            bins=[0, 15000, 30000, 50000, 80000, np.inf],
            labels=["Budget", "Value", "Mid-range", "Premium", "Luxury"],
            include_lowest=True,
        )
    if "age_years" in df.columns:
        df["age_band"] = pd.cut(
            df["age_years"],
            bins=[-0.1, 1, 3, 5, 8, 12, np.inf],
            labels=["0-1y", "1-3y", "3-5y", "5-8y", "8-12y", "12y+"],
        )
    if "mileage" in df.columns:
        df["mileage_band"] = pd.cut(
            df["mileage"],
            bins=[0, 10000, 30000, 60000, 100000, 150000, np.inf],
            labels=["0-10k", "10k-30k", "30k-60k", "60k-100k", "100k-150k", "150k+"],
            include_lowest=True,
        )
    return df


def summarize_segment_errors(df: pd.DataFrame, group_col: str, min_count: int = 30) -> pd.DataFrame:
    if group_col not in df.columns or "residual" not in df.columns or "price" not in df.columns:
        return pd.DataFrame()
    rep = (
        df.groupby(group_col, dropna=False, observed=True)
        .agg(
            n=("residual", "size"),
            mae=("residual", lambda r: float(np.mean(np.abs(r)))),
            rmse=("residual", lambda r: float(np.sqrt(np.mean(np.square(r))))),
            bias=("residual", "mean"),
            median_price=("price", "median"),
        )
        .reset_index()
    )
    rep["rmape"] = rep["mae"] / rep["median_price"].clip(lower=1.0)
    return rep[rep["n"] >= min_count].sort_values("rmse", ascending=False)

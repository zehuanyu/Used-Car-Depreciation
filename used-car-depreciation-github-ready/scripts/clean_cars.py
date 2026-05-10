import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "external" / "cars.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "cars_clean.parquet"
FILL_BOOL_BLANK_AS_FALSE = False
PRICE_MIN, PRICE_MAX = 1_000, 300_000
AGE_MAX_YEARS = 40

BOOL_COLS = ["fleet", "frame_damaged", "has_accidents"]
CAT_COLS = [
    "body_type",
    "fuel_type",
    "interior_color",
    "listing_color",
    "make_name",
    "model_name",
    "transmission",
    "trim_name",
    "wheel_system",
    "vehicle_damage_category",
    "engine_type",
]
NUM_COLS = ["price", "mileage", "horsepower", "engine_displacement", "owner_count", "daysonmarket", "year"]


def normalize_bool_col(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return s.astype("boolean")

    out = pd.Series(pd.NA, index=s.index, dtype="object")
    ss = s.astype(str).str.upper().str.strip()
    ss = ss.str.replace(r"\s+", " ", regex=True)
    token = ss.str.extract(
        r"(TRUE|FALSE|T|F|YES|NO|Y|N|UNKNOWN|NULL|NONE|NAN|NA|EMPTY|BLANK)",
        expand=False,
    )

    true_set = {"TRUE", "T", "YES", "Y"}
    false_set = {"FALSE", "F", "NO", "N"}

    out[token.isin(true_set)] = True
    out[token.isin(false_set)] = False
    return out.astype("boolean")


def parse_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("#", "", regex=False).str.strip()
    return pd.to_datetime(s, errors="coerce")


def disp_to_liters(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x > 25, x / 1000.0, x)


def extract_cyl_num(text):
    s = pd.Series(text).astype(str).str.upper()
    n = s.str.extract(r"(\d+)")
    return pd.to_numeric(n[0], errors="coerce")


def standardize_engine_type(df: pd.DataFrame) -> pd.Series:
    et = df["engine_type"].astype(str).str.strip().str.upper()
    et = et.replace({"NONE": "", "NAN": ""})
    need = et.eq("")

    ec = df.get("engine_cylinders", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    fuel = df.get("fuel_type", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()

    base = pd.Series(np.where(need, ec.str.extract(r"([IVHW]\d+)", expand=False), None), index=df.index)
    diesel = ec.str.contains("DIESEL", na=False) | fuel.str.contains("DIESEL", na=False)
    out = et.copy()
    out[need & base.notna() & diesel] = base[need & base.notna()] + " DIESEL"
    out[need & base.notna() & ~diesel] = base[need & base.notna()]
    out = out.replace({"": np.nan})
    return out


def layered_impute_numeric(df, col, keys_list):
    s = df[col].copy()
    na_mask = s.isna()

    for keys in keys_list:
        if not na_mask.any():
            break

        grp = (
            df.loc[~s.isna(), keys + [col]]
            .groupby(keys, dropna=False, observed=True, sort=False)[col]
            .median()
            .reset_index()
            .rename(columns={col: "__med"})
        )

        sub = df.loc[na_mask, keys].copy()
        sub["__rowid"] = sub.index
        sub = sub.merge(grp, on=keys, how="left")
        s.loc[sub["__rowid"]] = sub["__med"].to_numpy()
        na_mask = s.isna()

    return s


def layered_impute_mode(df, col, keys_list):
    s = df[col].copy()
    na_mask = s.isna()

    for keys in keys_list:
        if not na_mask.any():
            break

        grp = (
            df.loc[~s.isna(), keys + [col]]
            .groupby(keys, dropna=False, observed=True, sort=False)[col]
            .agg(lambda x: x.value_counts(dropna=True).idxmax() if len(x) else np.nan)
            .reset_index()
            .rename(columns={col: "__mode"})
        )

        sub = df.loc[na_mask, keys].copy()
        sub["__rowid"] = sub.index
        sub = sub.merge(grp, on=keys, how="left")
        s.loc[sub["__rowid"]] = sub["__mode"].to_numpy()
        na_mask = s.isna()

    return s


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]

    for c in set(CAT_COLS + NUM_COLS + BOOL_COLS + ["listed_date"]):
        if c not in df.columns:
            df[c] = pd.NA

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["listed_date"] = parse_date(df["listed_date"])
    df["age_years"] = (df["listed_date"].dt.year - df["year"]).astype("float32")

    for c in BOOL_COLS:
        df[c] = normalize_bool_col(df[c])
        df[c + "_unknown"] = df[c].isna()
        if FILL_BOOL_BLANK_AS_FALSE:
            df[c] = df[c].fillna(False)

    df["engine_disp_l"] = disp_to_liters(df["engine_displacement"]).astype("float32")
    df["engine_type_std"] = standardize_engine_type(df)
    df["cylinders_num"] = extract_cyl_num(df["engine_type_std"].fillna(df["engine_cylinders"])).astype("float32")
    df["disp_missing"] = df["engine_disp_l"].isna()
    df["etype_missing"] = df["engine_type_std"].isna()

    df = df[(df["price"] > PRICE_MIN) & (df["price"] < PRICE_MAX)]
    df = df[df["mileage"].fillna(0) >= 0]
    df = df[(df["age_years"].isna()) | ((df["age_years"] >= 0) & (df["age_years"] <= AGE_MAX_YEARS))]

    for c in set(
        CAT_COLS
        + [
            "engine_type_std",
            "body_type",
            "make_name",
            "model_name",
            "trim_name",
            "wheel_system",
            "fuel_type",
            "transmission",
            "listing_color",
            "interior_color",
        ]
    ):
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df.reset_index(drop=True)


def impute_powertrain(df: pd.DataFrame) -> pd.DataFrame:
    keys_layers = [
        ["make_name", "model_name", "year", "trim_name"],
        ["make_name", "model_name", "year"],
        ["make_name", "model_name"],
        ["make_name"],
        ["body_type"],
    ]

    df["engine_disp_l"] = layered_impute_numeric(df, "engine_disp_l", keys_layers)
    df["engine_type_std"] = layered_impute_mode(df, "engine_type_std", keys_layers)

    still_missing = df["engine_disp_l"].isna() & df["horsepower"].notna()
    try:
        from sklearn.linear_model import LinearRegression

        fit_mask = df["engine_disp_l"].notna() & df["horsepower"].notna()
        if fit_mask.sum() > 100 and still_missing.any():
            X = pd.DataFrame(
                {
                    "hp": df.loc[fit_mask, "horsepower"].astype("float32"),
                    "cyl": df.loc[fit_mask, "cylinders_num"].fillna(0).astype("float32"),
                    "year": df.loc[fit_mask, "year"].astype("float32"),
                }
            )
            y = df.loc[fit_mask, "engine_disp_l"].astype("float32")
            lr = LinearRegression().fit(X, y)

            Xmiss = pd.DataFrame(
                {
                    "hp": df.loc[still_missing, "horsepower"].astype("float32"),
                    "cyl": df.loc[still_missing, "cylinders_num"].fillna(0).astype("float32"),
                    "year": df.loc[still_missing, "year"].astype("float32"),
                }
            )
            df.loc[still_missing, "engine_disp_l"] = lr.predict(Xmiss).astype("float32")
    except Exception as exc:
        print(f"[warn] skip fallback regression for engine_disp_l: {exc}")

    drop_mask = df["engine_disp_l"].isna() & df["engine_type_std"].isna() & df["horsepower"].isna()
    before, after = len(df), len(df[~drop_mask])
    if before != after:
        print(f"[info] drop rows with too little powertrain info: {before - after}")

    return df[~drop_mask].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Clean the raw used-car dataset and save a processed parquet.")
    ap.add_argument("--input", type=str, default=str(INPUT_CSV), help="Path to the raw CSV file.")
    ap.add_argument("--output", type=str, default=str(OUTPUT_PARQUET), help="Path to the output parquet file.")
    args = ap.parse_args()

    input_csv = Path(args.input)
    output_parquet = Path(args.output)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    print("[info] reading csv ...")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"[info] raw shape: {df.shape}")

    df = clean_base(df)
    print(f"[info] after base clean: {df.shape}")
    df = impute_powertrain(df)
    df.to_parquet(output_parquet, index=False)

    miss_disp = df["engine_disp_l"].isna().mean() if "engine_disp_l" in df else np.nan
    miss_type = df["engine_type_std"].isna().mean() if "engine_type_std" in df else np.nan
    print(f"[done] saved -> {output_parquet}")
    print(f"[stats] remain rows: {len(df):,}")
    print(f"[stats] missing engine_disp_l: {miss_disp:.2%}")
    print(f"[stats] missing engine_type_std: {miss_type:.2%}")


if __name__ == "__main__":
    main()

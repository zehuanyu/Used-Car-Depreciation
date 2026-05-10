import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "cars_clean.parquet"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "figures"
ONLY_USED = True


def sample_vals(s: pd.Series, k=3):
    vals = s.dropna().unique()[:k]
    return "; ".join(map(lambda x: str(x)[:40], vals))


def main():
    ap = argparse.ArgumentParser(description="Generate EDA charts and a data dictionary from a cleaned parquet dataset.")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to the cleaned parquet dataset.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output directory for report assets.")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    df = df[df["price"].notna()].copy()
    if ONLY_USED:
        used_mask = (df.get("mileage", 0).fillna(0) > 0) | (df.get("owner_count", 0).fillna(0) >= 1)
        df = df[used_mask].copy()

    meta = []
    for c in df.columns:
        meta.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "missing_rate": float(df[c].isna().mean()),
                "nunique": int(df[c].nunique(dropna=True)),
                "examples": sample_vals(df[c]),
            }
        )
    pd.DataFrame(meta).sort_values(["dtype", "missing_rate", "nunique", "column"]).to_csv(
        out_dir / "data_dictionary.csv", index=False, encoding="utf-8-sig"
    )

    price = pd.to_numeric(df["price"], errors="coerce").dropna()
    qs = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0]
    edges = price.quantile(qs).to_numpy()
    step = 1000 if price.max() <= 100_000 else 5_000
    edges = np.array([int(np.round(x / step) * step) for x in edges])
    edges[0] = max(0, edges[0])
    edges[-1] = edges[-1] + step
    edges = np.unique(edges)

    labels = [f"{int(a / 1000)}k-{int(b / 1000)}k" for a, b in zip(edges[:-1], edges[1:])]
    cats = pd.cut(price, bins=edges, labels=labels, right=False, include_lowest=True)
    counts = cats.value_counts().reindex(labels).fillna(0).astype(int)
    cum = (counts / counts.sum()).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, cum.values * 100, marker="o")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Vehicle count")
    ax.set_title("Price distribution (quantile bins + cumulative share)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "price_distribution_pareto.png", dpi=150)
    plt.close(fig)

    if "age_years" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = df[["age_years", "price"]].dropna()
        ax.hexbin(sub["age_years"], sub["price"], gridsize=40, bins="log")
        bins = np.arange(0, min(40, sub["age_years"].max() + 0.5), 0.5)
        med = sub.groupby(pd.cut(sub["age_years"], bins=bins, right=False), observed=True, sort=False)["price"].median()
        if len(med) > 0:
            centers = np.array([(iv.left + iv.right) / 2 for iv in med.index])
            ax.plot(centers, med.values)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Price vs Age (density + median line)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "price_vs_age_hexbin.png", dpi=150)
        plt.close(fig)

    if "mileage" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = df[["mileage", "price"]].dropna().copy()
        sub["mileage"] = sub["mileage"].clip(lower=1)
        sub = sub[(sub["mileage"] >= 500) & (sub["mileage"] <= 300_000)]
        p_cap = np.quantile(sub["price"], 0.995)
        sub["price"] = sub["price"].clip(upper=p_cap)
        sub["log_mileage"] = np.log10(sub["mileage"])
        ax.hexbin(
            sub["log_mileage"].to_numpy(),
            sub["price"].to_numpy(),
            gridsize=(70, 40),
            bins="log",
            mincnt=20,
            extent=[float(sub["log_mileage"].min()), float(sub["log_mileage"].max()), 0.0, float(p_cap)],
        )

        xbins = np.logspace(np.log10(sub["mileage"].min()), np.log10(sub["mileage"].max()), 45)
        grp = sub.groupby(pd.cut(sub["mileage"], bins=xbins, include_lowest=True), observed=True)
        stats = grp["price"].quantile([0.25, 0.5, 0.75]).unstack()
        counts = grp.size()
        stats = stats[counts >= 150]

        if not stats.empty:
            centers_mi = np.array([(iv.left * iv.right) ** 0.5 for iv in stats.index])
            centers_log = np.log10(centers_mi)

            def smooth(y):
                return pd.Series(y).rolling(3, center=True, min_periods=1).median().to_numpy()

            ax.plot(centers_log, smooth(stats[0.50].to_numpy()))
            ax.plot(centers_log, smooth(stats[0.25].to_numpy()))
            ax.plot(centers_log, smooth(stats[0.75].to_numpy()))

        xticks = np.arange(int(np.floor(sub["log_mileage"].min())), int(np.ceil(sub["log_mileage"].max())) + 1)

        def fmt_tick(p):
            v = 10 ** p
            if v >= 1_000_000:
                return f"{int(v / 1_000_000)}M"
            if v >= 1_000:
                return f"{int(v / 1_000)}k"
            return f"{int(v)}"

        ax.set_xticks(xticks)
        ax.set_xticklabels([fmt_tick(p) for p in xticks])
        ax.set_xlabel("Mileage (log scale)")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Price vs Mileage (density + quantile lines)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "price_vs_mileage_hexbin.png", dpi=150)
        plt.close(fig)

    if "body_type" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = df[["body_type", "price"]].dropna()
        top = sub["body_type"].value_counts().head(10).index
        ax.boxplot([sub.loc[sub["body_type"] == bt, "price"].values for bt in top], showfliers=False)
        ax.set_xticklabels(top, rotation=20, ha="right")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Price by body type (boxplot)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "price_by_body_type_box.png", dpi=150)
        plt.close(fig)

    if "make_name" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        vc = df["make_name"].value_counts().head(20)
        x = np.arange(len(vc))
        bars = ax.bar(x, vc.values)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(vc.index, rotation=25, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Top-20 brands by count")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "make_count_top20.png", dpi=150)
        plt.close(fig)

    if "listed_date" in df.columns:
        sub = df[["listed_date", "price"]].dropna().copy()
        sub["month"] = sub["listed_date"].dt.to_period("M").dt.to_timestamp()
        ts = sub.groupby("month", observed=True, sort=False)["price"].median().sort_index()
        counts = sub.groupby("month").size()
        ts = ts[counts[ts.index] >= 500]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts.index, ts.values, marker="o", markersize=3, linewidth=1)
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Month")
        ax.set_ylabel("Median price (USD)")
        ax.set_title("Monthly Median Price Trend")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / "median_price_by_month_en.png", dpi=150)
        plt.close(fig)

    print("[done] charts saved to:", out_dir)


if __name__ == "__main__":
    main()

from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_price_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "cars_clean_v2_sample.csv"
BENCHMARK_PATH = PROJECT_ROOT / "results" / "benchmarks" / "benchmark_results.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
FEATURES_PATH = PROJECT_ROOT / "results" / "feature_importance.csv"

st.set_page_config(page_title="Used Car Depreciation Lab", layout="wide")


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df["value_per_year"] = df["price"] / np.maximum(df["age_years"], 0.5)
    return df


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_feature_importance():
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return None


@st.cache_data
def load_benchmarks():
    if BENCHMARK_PATH.exists():
        return pd.read_csv(BENCHMARK_PATH)
    return None


@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def map_binary(value: str):
    if value == "Yes":
        return 1
    if value == "No":
        return 0
    return np.nan


def valuation_label(delta_ratio: float) -> tuple[str, str]:
    if delta_ratio <= -0.08:
        return "Potential Bargain", "#0f766e"
    if delta_ratio >= 0.08:
        return "Priced Aggressively", "#b45309"
    return "Near Model Estimate", "#475569"


def build_row_from_inputs(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mileage": inputs["mileage"],
                "log_mileage": np.log10(max(1.0, inputs["mileage"])),
                "age_years": inputs["age_years"],
                "age_sq": inputs["age_years"] ** 2,
                "engine_disp_l": inputs["engine_disp_l"],
                "cylinders_num": inputs["cylinders_num"] if inputs["cylinders_num"] != "" else np.nan,
                "has_accidents": map_binary(inputs["has_accidents"]),
                "frame_damaged": map_binary(inputs["frame_damaged"]),
                "fleet": 0,
                "make_name": inputs["make_name"] if inputs["make_name"] else np.nan,
                "model_name": inputs["model_name"] if inputs["model_name"] else np.nan,
                "body_type": inputs["body_type"] if inputs["body_type"] else np.nan,
                "fuel_type": inputs["fuel_type"] if inputs["fuel_type"] else np.nan,
                "transmission": inputs["transmission"] if inputs["transmission"] else np.nan,
                "wheel_system": inputs["wheel_system"] if inputs["wheel_system"] else np.nan,
                "trim_name": inputs["trim_name"] if inputs["trim_name"] else np.nan,
                "engine_type_std": inputs["engine_type_std"] if inputs["engine_type_std"] else np.nan,
            }
        ]
    )


def predict_price(bundle, row: pd.DataFrame) -> float:
    for col in bundle["cats"]:
        if col in row.columns:
            row[col] = row[col].astype("category")
    pred_log1p = bundle["model"].predict(row[bundle["features"]])[0]
    return float(np.exp(pred_log1p) - 1)


def predict_curve(bundle, base_row: dict, sweep="mileage"):
    rows = []
    if sweep == "mileage":
        xs = np.logspace(np.log10(1000), np.log10(200000), 36)
        label = "Mileage"
        for x in xs:
            r = base_row.copy()
            r["mileage"] = float(x)
            r["log_mileage"] = float(np.log10(max(1.0, x)))
            rows.append(r)
    else:
        xs = np.linspace(0, 15, 36)
        label = "Age (years)"
        for x in xs:
            r = base_row.copy()
            r["age_years"] = float(x)
            r["age_sq"] = float(x**2)
            rows.append(r)

    curve_df = pd.DataFrame(rows)
    for col in bundle["cats"]:
        if col in curve_df.columns:
            curve_df[col] = curve_df[col].astype("category")
    pred = np.exp(bundle["model"].predict(curve_df[bundle["features"]])) - 1
    return pd.DataFrame({label: xs, "Predicted Price": pred})


def find_comparables(df: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    sub = df.copy()
    for col in ["make_name", "model_name", "body_type"]:
        val = inputs.get(col)
        if val:
            matched = sub[sub[col] == val]
            if len(matched) >= 5:
                sub = matched

    if sub.empty:
        return sub

    sub = sub.copy()
    sub["distance"] = (
        (sub["mileage"] - inputs["mileage"]).abs() / 20000.0
        + (sub["age_years"] - inputs["age_years"]).abs() / 3.0
        + (sub["engine_disp_l"] - inputs["engine_disp_l"]).abs() / 1.5
    )
    cols = ["price", "make_name", "model_name", "trim_name", "body_type", "mileage", "age_years", "distance"]
    return sub.sort_values("distance").head(8)[cols]


def curve_chart(df: pd.DataFrame, x_col: str, color: str):
    return (
        alt.Chart(df)
        .mark_line(point=True, color=color, strokeWidth=3)
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y("Predicted Price", title="Predicted Price (USD)"),
            tooltip=[x_col, alt.Tooltip("Predicted Price", format=",.0f")],
        )
        .properties(height=280)
    )


def make_overview_chart(df: pd.DataFrame):
    chart_df = df[["age_years", "price"]].dropna().copy()
    chart_df["price_cap"] = chart_df["price"].clip(upper=chart_df["price"].quantile(0.98))
    return (
        alt.Chart(chart_df)
        .mark_circle(size=45, opacity=0.3, color="#0f766e")
        .encode(
            x=alt.X("age_years", title="Vehicle Age (years)"),
            y=alt.Y("price_cap", title="Price (USD, capped at 98th percentile)"),
            tooltip=[alt.Tooltip("age_years", format=".1f"), alt.Tooltip("price_cap", format=",.0f")],
        )
        .properties(height=340)
    )


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #22252f 0%, #1c2029 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {
        border-radius: 12px;
    }
    .section-eyebrow {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: #0f766e;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .hero-card {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #f4efe4 0%, #ebe3d1 100%);
        border: 1px solid rgba(15, 118, 110, 0.18);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        color: #18212f;
    }
    .signal-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: linear-gradient(180deg, #fffdfa 0%, #f5f1e8 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        color: #18212f;
    }
    .hero-card h3, .hero-card p, .signal-card strong, .signal-card span, .signal-card div {
        color: #18212f !important;
    }
    .signal-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #475569 !important;
        margin-bottom: 0.35rem;
    }
    .signal-metric {
        font-size: 1rem;
        line-height: 1.65;
        color: #18212f !important;
    }
    .insight-note {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: rgba(15, 118, 110, 0.08);
        border: 1px solid rgba(15, 118, 110, 0.15);
    }
    [data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Used Car Depreciation Lab")
st.caption("A portfolio-focused app for valuation, depreciation behavior, and model insight.")

if not DATA_PATH.exists():
    st.error("Sample dataset not found in `data/sample/`.")
    st.stop()

df = load_reference_data()
metrics = load_metrics()
feature_importance = load_feature_importance()
benchmarks = load_benchmarks()
bundle = load_model_bundle()

hero_left, hero_right = st.columns([1.35, 1])
with hero_left:
    st.markdown(
        """
        <div class="hero-card">
        <h3 style="margin-top:0;">What this app does</h3>
        <p style="margin-bottom:0;">
        Estimate used-car value from listing attributes, visualize how price changes with mileage and age,
        and connect the prediction back to data-driven market context.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    if metrics:
        st.markdown(
            f"""
            <div class="signal-card">
            <div class="signal-kicker">Model Snapshot</div>
            <div class="signal-metric">
            <strong>MAE:</strong> ${metrics['MAE']:,.0f}<br>
            <strong>RMSE:</strong> ${metrics['RMSE']:,.0f}<br>
            <strong>R²:</strong> {metrics['R2']:.3f}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

overview_tab, valuation_tab, insight_tab = st.tabs(["Overview", "Valuation Workbench", "Model Insights"])

with overview_tab:
    st.markdown("<div class='section-eyebrow'>Market View</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Listings", f"{len(df):,}")
    c2.metric("Brands", f"{df['make_name'].nunique():,}")
    c3.metric("Median Price", f"${df['price'].median():,.0f}")
    c4.metric("Median Mileage", f"{df['mileage'].median():,.0f}")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Price vs. Vehicle Age")
        st.altair_chart(make_overview_chart(df), use_container_width=True)
    with right:
        st.subheader("Project Framing")
        st.write(
            "This project studies how vehicle age, mileage, and configuration shape resale value. "
            "The goal is not only to predict price, but also to understand depreciation patterns and model limitations."
        )
        top_makes = df["make_name"].value_counts().head(10).rename_axis("Make").reset_index(name="Listings")
        st.bar_chart(top_makes.set_index("Make"))

with valuation_tab:
    st.markdown("<div class='section-eyebrow'>Interactive Valuation</div>", unsafe_allow_html=True)
    if bundle is None:
        st.info("Add `lgbm_price_model.joblib` to `models/` to enable live valuation.")
    else:
        make_to_models = df.groupby("make_name")["model_name"].unique().apply(sorted).to_dict()
        make_model_to_body = (
            df.groupby(["make_name", "model_name"])["body_type"].unique().apply(lambda arr: sorted([str(x) for x in arr])).to_dict()
        )
        make_model_body_to_trim = (
            df.groupby(["make_name", "model_name", "body_type"])["trim_name"].unique().apply(lambda arr: sorted([str(x) for x in arr])).to_dict()
        )

        defaults = {
            "make_name": "Toyota" if "Toyota" in set(df["make_name"]) else sorted(df["make_name"].dropna().unique())[0],
        }
        default_models = list(make_to_models.get(defaults["make_name"], []))
        defaults["model_name"] = "Camry" if "Camry" in default_models else (default_models[0] if default_models else "")
        default_bodies = list(make_model_to_body.get((defaults["make_name"], defaults["model_name"]), []))
        defaults["body_type"] = default_bodies[0] if default_bodies else ""
        default_trims = list(make_model_body_to_trim.get((defaults["make_name"], defaults["model_name"], defaults["body_type"]), []))
        defaults["trim_name"] = default_trims[0] if default_trims else ""

        with st.sidebar:
            st.header("Vehicle Inputs")
            make_name = st.selectbox("Make", sorted(make_to_models.keys()), index=sorted(make_to_models.keys()).index(defaults["make_name"]))
            model_options = list(make_to_models.get(make_name, []))
            model_name = st.selectbox("Model", model_options, index=model_options.index(defaults["model_name"]) if defaults["model_name"] in model_options else 0)
            body_options = list(make_model_to_body.get((make_name, model_name), [])) or [""]
            body_type = st.selectbox("Body Type", body_options, index=0)
            trim_options = list(make_model_body_to_trim.get((make_name, model_name, body_type), [])) or [""]
            trim_name = st.selectbox("Trim", trim_options, index=0)
            fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].dropna().unique()))
            transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
            wheel_system = st.selectbox("Wheel System", sorted(df["wheel_system"].dropna().unique()))
            engine_type_std = st.selectbox("Engine Type", sorted(df["engine_type_std"].dropna().unique()))
            mileage = st.slider("Mileage", min_value=0, max_value=220000, value=30000, step=1000)
            age_years = st.slider("Age (years)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
            engine_disp_l = st.slider("Engine displacement (L)", min_value=0.8, max_value=8.0, value=2.5, step=0.1)
            cylinders_num = st.selectbox("Cylinders", [2, 3, 4, 5, 6, 8, 10, 12], index=2)
            asking_price = st.number_input("Optional asking price", min_value=0.0, value=28000.0, step=500.0)
            has_accidents = st.selectbox("Has Accidents?", ["No", "Yes"], index=0)
            frame_damaged = st.selectbox("Frame Damaged?", ["No", "Yes"], index=0)
            st.caption("Tip: adjust asking price to see whether the listing looks fairly priced relative to the model estimate.")

        inputs = {
            "make_name": make_name,
            "model_name": model_name,
            "body_type": body_type,
            "trim_name": trim_name,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "wheel_system": wheel_system,
            "engine_type_std": engine_type_std,
            "mileage": mileage,
            "age_years": age_years,
            "engine_disp_l": engine_disp_l,
            "cylinders_num": cylinders_num,
            "asking_price": asking_price,
            "has_accidents": has_accidents,
            "frame_damaged": frame_damaged,
        }

        row = build_row_from_inputs(inputs)
        predicted_price = predict_price(bundle, row.copy())
        delta = asking_price - predicted_price
        delta_ratio = delta / max(predicted_price, 1.0)
        badge, badge_color = valuation_label(delta_ratio)
        comparables = find_comparables(df, inputs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Estimate", f"${predicted_price:,.0f}")
        m2.metric("Asking Price", f"${asking_price:,.0f}")
        m3.metric("Difference", f"${delta:,.0f}", delta=f"{delta_ratio:.1%}")
        m4.markdown(
            f"""
            <div class='signal-card'>
            <div class='signal-kicker'>Valuation Signal</div>
            <div class='signal-metric'><strong style='color:{badge_color} !important;'>{badge}</strong><br>
            <span>Model-based pricing interpretation</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        curve_source = row.iloc[0].to_dict()
        mileage_curve = predict_curve(bundle, curve_source, sweep="mileage")
        age_curve = predict_curve(bundle, curve_source, sweep="age")

        left, right = st.columns(2)
        with left:
            st.subheader("Depreciation by Mileage")
            st.altair_chart(curve_chart(mileage_curve, "Mileage", "#0f766e"), use_container_width=True)
        with right:
            st.subheader("Depreciation by Age")
            st.altair_chart(curve_chart(age_curve, "Age (years)", "#b45309"), use_container_width=True)

        st.markdown(
            """
            <div class="insight-note">
            The valuation signal compares the asking price to the model estimate. It is most useful as a directional check,
            not a definitive appraisal.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Nearest Comparable Listings in the Sample")
        if comparables.empty:
            st.info("Not enough comparable listings were found in the sample data.")
        else:
            st.dataframe(comparables, use_container_width=True)

with insight_tab:
    st.markdown("<div class='section-eyebrow'>Model Readout</div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Feature Importance")
        if feature_importance is not None:
            top_features = feature_importance.head(12).copy()
            feat_chart = (
                alt.Chart(top_features)
                .mark_bar(color="#0f766e")
                .encode(
                    x=alt.X("gain", title="Gain"),
                    y=alt.Y("feature", sort="-x", title="Feature"),
                    tooltip=["feature", alt.Tooltip("gain", format=",.0f")],
                )
                .properties(height=360)
            )
            st.altair_chart(feat_chart, use_container_width=True)
        else:
            st.info("Feature importance file not found.")

    with right:
        st.subheader("Benchmark Models")
        if benchmarks is not None:
            st.dataframe(benchmarks, use_container_width=True)
        else:
            st.info("Run `scripts/benchmark_models.py` to populate benchmark results.")

    st.subheader("Interpretation")
    st.write(
        "The model currently indicates that age and mileage dominate depreciation, while trim, model, and engine setup "
        "provide meaningful secondary pricing signal. Benchmarking and segment-level error analysis help test whether "
        "that performance holds across different market slices."
    )

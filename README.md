# Used Car Depreciation

A Python machine learning project for analyzing used-car depreciation patterns and estimating resale prices from listing-level vehicle attributes.

This project focuses on building a practical valuation workflow: cleaning messy used-car listing data, engineering depreciation-related features, training predictive models, benchmarking performance, analyzing model errors by segment, and packaging the result into a Streamlit app.

---

## Overview

Used-car prices are affected by many interacting factors, including age, mileage, make, model, trim, body type, fuel type, transmission, accident history, and market timing.

This project was built to answer a practical question:

> Can we estimate a fair used-car resale price while also understanding the major drivers of depreciation?

The final workflow includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- LightGBM-based price modeling
- Benchmark comparison against baseline models
- Segment-level error analysis
- Interactive Streamlit valuation app

---

## Project Workflow

```text
Raw vehicle listings
        ↓
Data cleaning and preprocessing
        ↓
Feature engineering
        ↓
Exploratory data analysis
        ↓
Model training and benchmarking
        ↓
Error analysis by market segment
        ↓
Streamlit valuation app
```

---

## Key Features

- Cleans and standardizes structured used-car listing data
- Derives depreciation-focused features such as vehicle age, mileage transforms, and engine displacement
- Trains a LightGBM regression model for price prediction
- Benchmarks model performance against simpler baselines
- Evaluates prediction error by make, body type, age band, mileage band, and price band
- Includes a Streamlit app for interactive vehicle price estimation
- Provides reusable scripts for cleaning, training, scoring, benchmarking, and error analysis

---

## Dataset

The project uses used-vehicle listing data with features such as:

- Price
- Model year
- Listing date
- Mileage
- Make and model
- Trim
- Body type
- Fuel type
- Transmission
- Wheel system
- Engine characteristics
- Accident and fleet history indicators

The repository includes sample data and lightweight artifacts for demonstration. The full original raw dataset is not included.

---

## Methodology

### 1. Data Cleaning

The cleaning pipeline prepares raw listing data for modeling by:

- Standardizing column names
- Converting mixed-type numeric fields
- Normalizing boolean fields such as accident and fleet history
- Converting engine displacement into liter-based values
- Deriving vehicle age from listing date and model year
- Handling missing or inconsistent categorical values
- Creating processed and sample datasets for reproducible analysis

Main script:

```text
scripts/clean_cars.py
```

---

### 2. Exploratory Data Analysis

EDA was used to validate whether the data followed realistic market behavior.

The analysis focuses on:

- Price distribution
- Price vs. mileage
- Price vs. vehicle age
- Brand and body-type patterns
- Monthly price trends
- Depreciation behavior across vehicle segments

Generated figures are stored in:

```text
reports/figures/
```

---

### 3. Feature Engineering

The model uses both raw listing attributes and engineered features, including:

- `age_years`
- `age_sq`
- `log_mileage`
- `engine_disp_l`
- `hp_per_l`
- Standardized engine type labels
- Normalized boolean indicators

These features help capture nonlinear depreciation behavior and vehicle configuration effects.

---

### 4. Modeling

The main predictive model is a LightGBM regressor trained on tabular vehicle data.

LightGBM was selected because:

- The dataset is structured and feature-rich
- Price relationships are nonlinear
- Numeric and categorical variables interact in complex ways
- Gradient boosting performs well on tabular regression problems

Benchmark models include:

- Median baseline
- Linear regression
- Gradient boosting
- Random forest

Benchmark script:

```text
scripts/benchmark_models.py
```

---

## Results

Historical full-run model performance:

| Metric | Result |
|---|---:|
| MAE | about $1.7k |
| RMSE | about $2.5k |
| R² | about 0.97 |

The included sample benchmark shows that tree-based models perform strongly compared with a simple median baseline, while linear regression remains useful as an interpretable reference.

Result files:

```text
results/metrics.json
results/benchmarks/benchmark_results.csv
results/error_analysis/
```

---

## Streamlit App

The repository includes a Streamlit app that turns the model into an interactive valuation tool.

The app supports:

- Project overview
- Vehicle input form
- Estimated resale price
- Asking price comparison
- Depreciation curves by age and mileage
- Comparable sample listings
- Model insight and benchmark summaries

App file:

```text
app/streamlit_app.py
```

---

## Repository Structure

```text
Used-Car-Depreciation/
├── app/                         # Streamlit web app
├── data/
│   ├── external/                # Raw data placeholder
│   ├── processed/               # Processed data placeholder
│   └── sample/                  # Sample cleaned dataset
├── docs/                        # Case study and roadmap
├── models/                      # Trained model bundle
├── reports/
│   ├── figures/                 # EDA figures
│   └── used_car_depreciation_report.pdf
├── results/
│   ├── benchmarks/              # Benchmark results
│   ├── error_analysis/          # Segment-level error analysis
│   └── sample_scored/           # Scored sample outputs
├── scripts/                     # Data, modeling, and analysis scripts
├── src/used_car_project/        # Reusable project utilities
├── INTERVIEW_GUIDE.md
├── PORTFOLIO_SUMMARY.md
├── RUN_APP.bat
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/zehuanyu/Used-Car-Depreciation.git
cd Used-Car-Depreciation
```

### 2. Create environment and install dependencies

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

Option A: double-click

```text
RUN_APP.bat
```

Option B: run manually

```bash
py -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

## Reproduce the Workflow

Clean raw data:

```bash
py scripts/clean_cars.py --input data/external/cars.csv --output data/processed/cars_clean.parquet
```

Generate EDA assets:

```bash
py scripts/dataset_report.py --data data/processed/cars_clean.parquet --out reports/figures
```

Train the main model:

```bash
py scripts/train_price_lgbm_from_v2.py --data data/processed/cars_clean_v2.csv --out results/artifacts
```

Benchmark models:

```bash
py scripts/benchmark_models.py --data data/sample/cars_clean_v2_sample.csv --out results/benchmarks --row-cap 1500 --n-jobs 1
```

Run error analysis:

```bash
py scripts/error_analysis.py --scored results/sample_scored/scored_all.csv --out results/error_analysis
```

---

## Key Takeaways

- Data cleaning quality was as important as model choice.
- Vehicle age and mileage were the strongest depreciation drivers.
- Vehicle configuration added meaningful secondary pricing signal.
- Benchmarking made the model performance easier to interpret.
- Segment-level error analysis helped identify where the model is more or less reliable.
- The Streamlit app makes the project easier to demonstrate as a practical valuation tool.

---

## Future Improvements

- Add SHAP-based feature interpretation
- Improve comparable-listing logic in the Streamlit app
- Add uncertainty estimates for predicted prices
- Expand location-based market features
- Retrain with a larger curated dataset
- Add automated model evaluation reports

---

## Tech Stack

Python, Pandas, LightGBM, Scikit-learn, Streamlit, Matplotlib, SQL-style EDA, Machine Learning, Data Cleaning, Regression Modeling

---

## Author

Zehuan Yu

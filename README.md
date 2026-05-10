# Used Car Depreciation

An end-to-end data analytics and machine learning project that studies how used vehicles lose value over time and predicts resale price from listing-level vehicle attributes.

## Executive Summary

This project began as a used-car price prediction study and was later reorganized into a mature, shareable data project. The final version combines:

- structured data cleaning
- feature engineering
- exploratory data analysis
- machine learning for price prediction
- benchmark comparison against simpler baselines
- segment-level error analysis
- a Streamlit web application for interactive valuation

The central business question is:

**How do age, mileage, and vehicle characteristics drive used-car depreciation, and how accurately can we estimate a fair resale value from listing data?**

## Why This Project Matters

Used-car pricing is noisy and hard to evaluate consistently. Buyers want to know whether a listing is overpriced. Sellers want a realistic market estimate. Analysts want to understand which attributes contribute most to depreciation.

This project addresses those needs in two ways:

1. It builds a predictive model that estimates vehicle price.
2. It analyzes the structure of depreciation so the result is interpretable, not only predictive.

## What I Did

### 1. Data preparation

I started by cleaning and standardizing the vehicle listing dataset:

- normalized column names
- converted numeric fields such as mileage, year, and price
- standardized boolean features such as accident history and fleet usage
- derived model-friendly fields such as `age_years`, `age_sq`, `log_mileage`, and `engine_disp_l`
- handled missing and inconsistent vehicle attributes
- organized the project into raw, processed, sample, model, and result layers

### 2. Exploratory analysis

I used EDA to understand price behavior before modeling:

- price distribution across the market
- price vs. mileage
- price vs. vehicle age
- differences across body types and brands
- monthly median price trend

These artifacts are saved in [reports/figures](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/reports/figures).

### 3. Feature engineering

To improve model performance and interpretability, I created features such as:

- `age_years`
- `age_sq`
- `log_mileage`
- `hp_per_l`
- normalized engine displacement
- standardized categorical vehicle descriptors

### 4. Predictive modeling

I trained a LightGBM regressor for used-car price estimation and kept the workflow reproducible through scripts instead of one-off notebook logic.

I also added a benchmark layer so the project does not rely on a single model without context. Included benchmark outputs compare:

- Median Baseline
- Linear Regression
- Gradient Boosting
- Random Forest

### 5. Error analysis

To make the work more production-minded, I added segment-level evaluation instead of only headline metrics:

- error by make
- error by body type
- error by price band
- error by age band
- error by mileage band
- largest individual prediction misses

These are available in [results/error_analysis](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/error_analysis).

### 6. Web application

I built a Streamlit app so the project is usable by others, not just readable as code. The app supports:

- interactive vehicle input
- model-estimated price
- comparison to an asking price
- pricing signal interpretation
- depreciation curves by mileage and age
- comparable sample listings
- feature importance and benchmark summaries

## Current Results

### Historical full-run model performance

- MAE: about **$1.7k**
- RMSE: about **$2.5k**
- R²: about **0.97**

### Included sample benchmark result

On the included benchmark sample, the current saved comparison shows:

- **Random Forest** performed best
- **Linear Regression** provided a strong interpretable baseline
- **Gradient Boosting** was competitive
- **Median Baseline** performed far worse, confirming the problem is meaningfully learnable

See:

- [results/metrics.json](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/metrics.json)
- [results/benchmarks/benchmark_results.csv](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/benchmarks/benchmark_results.csv)

## Repository Structure

```text
used-car-depreciation-github-ready/
|-- app/                         # Streamlit web app
|-- data/
|   |-- external/                # raw data placeholder
|   |-- processed/               # processed data placeholder
|   `-- sample/                  # sample cleaned dataset
|-- docs/                        # case study and roadmap
|-- models/                      # trained model bundle
|-- reports/
|   |-- figures/                 # EDA outputs
|   `-- used_car_depreciation_report.pdf
|-- results/
|   |-- benchmarks/
|   |-- error_analysis/
|   `-- sample_scored/
|-- scripts/                     # reproducible project scripts
|-- src/used_car_project/        # reusable Python utilities
|-- INTERVIEW_GUIDE.md
|-- PORTFOLIO_SUMMARY.md
|-- RUN_APP.bat
|-- requirements.txt
`-- README.md
```

## How To Run The Project

### Install dependencies

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Open the web app

#### Option A: Double-click the launcher

```text
RUN_APP.bat
```

#### Option B: Run manually

```bash
py -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Reproducible Commands

### Clean raw data

```bash
py scripts/clean_cars.py --input data/external/cars.csv --output data/processed/cars_clean.parquet
```

### Generate EDA assets

```bash
py scripts/dataset_report.py --data data/processed/cars_clean.parquet --out reports/figures
```

### Train a validation model

```bash
py scripts/train_price_lgbm_from_v2.py --data data/processed/cars_clean_v2.csv --out results/artifacts
```

### Train on all available data

```bash
py scripts/final_train_on_all.py --data data/processed/cars_clean_v2.csv --out results/artifacts_full
```

### Benchmark alternative models

```bash
py scripts/benchmark_models.py --data data/sample/cars_clean_v2_sample.csv --out results/benchmarks --row-cap 1500 --n-jobs 1
```

### Score a dataset

```bash
py scripts/score_csv.py --data data/sample/cars_clean_v2_sample.csv --out results/sample_scored --segments make_name,body_type
```

### Run error analysis

```bash
py scripts/error_analysis.py --scored results/sample_scored/scored_all.csv --out results/error_analysis
```

## How I Would Present This In A Data Analyst Interview

I would describe the project like this:

> I built a used-car depreciation analysis and price prediction project to understand how vehicle age, mileage, and configuration affect resale value. I started by cleaning a messy listing dataset, standardizing vehicle attributes, and engineering features such as vehicle age, mileage transforms, and engine descriptors. Then I used EDA to validate the economic logic of the data, trained a machine learning model for valuation, compared it against simpler baselines, and added segment-level error analysis so the results were interpretable and operationally useful. Finally, I wrapped the work in a Streamlit application so the model could be used interactively by non-technical users.

## What An Interviewer Might Ask

### Why did you choose this project?

Because used-car pricing is a practical business problem with both analytical and modeling depth. It allows me to show data cleaning, feature engineering, exploratory analysis, predictive modeling, and communication in one project.

### What was the hardest part?

The hardest part was not training the model itself, but making the dataset consistent enough for reliable analysis. Vehicle listings contain noisy categorical values, mixed data types, and incomplete fields, so preprocessing quality directly affected model quality.

### Why did you use LightGBM?

Because the problem contains nonlinear relationships and many structured tabular features. LightGBM is a strong fit for tabular regression, handles mixed feature types well, and performs efficiently at scale.

### How did you validate the model?

I used a time-based validation split when listing dates were available, rather than relying only on random splitting. I also compared the main model against simpler baselines and looked beyond overall error into segment-level performance.

### What did you learn from the analysis?

The strongest price drivers were age and mileage, while model, trim, and engine configuration added important secondary signal. I also learned that good overall metrics do not guarantee uniform performance across all segments, which is why I added grouped error analysis.

### If you had more time, what would you improve?

- add SHAP-based interpretability
- expand comparable-market logic in the app
- retrain on a larger curated processed dataset inside the new repo structure
- add more robust validation and possibly location-based effects if available

## Best Files To Open First

- [PORTFOLIO_SUMMARY.md](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/PORTFOLIO_SUMMARY.md)
- [INTERVIEW_GUIDE.md](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/INTERVIEW_GUIDE.md)
- [app/streamlit_app.py](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/app/streamlit_app.py)
- [results/benchmarks/benchmark_results.csv](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/benchmarks/benchmark_results.csv)
- [reports/used_car_depreciation_report.pdf](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/reports/used_car_depreciation_report.pdf)

## Notes For Other Users

- This repository is usable immediately with the included sample data and trained model.
- Large raw source files are not included because they are too large for a standard GitHub repository.
- You can still rerun the workflow on your own data by placing files in `data/external/`.

## License

This project is released under the MIT License. See [LICENSE](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/LICENSE).

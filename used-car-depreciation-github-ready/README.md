# Used Car Depreciation

An end-to-end data science project that analyzes the drivers of used-car depreciation and builds machine learning models to estimate resale value from listing-level vehicle attributes.

## Overview

This repository is designed as a mature data project rather than a single modeling notebook. It combines:

- data cleaning and preprocessing
- exploratory data analysis
- predictive modeling
- model benchmarking
- segment-level error analysis
- an interactive Streamlit web application

The project is centered around two main questions:

1. Which factors drive used-car depreciation most strongly?
2. How accurately can we estimate market price from listing-level features?

## Main Features

- Cleaned and structured project layout suitable for GitHub
- Reusable Python scripts for training, scoring, benchmarking, and analysis
- Included sample dataset for reproducibility and demonstration
- Included trained model bundle for local demo use
- Included Streamlit app for interactive valuation and depreciation exploration
- Included EDA figures, report assets, and benchmark outputs

## Repository Structure

```text
used-car-depreciation/
|-- app/                         # Streamlit web app
|-- data/
|   |-- external/                # raw data placeholder (not tracked)
|   |-- processed/               # processed data placeholder
|   `-- sample/                  # sample cleaned dataset
|-- docs/                        # case study and roadmap
|-- models/                      # trained model bundle
|-- reports/
|   |-- figures/                 # EDA outputs
|   `-- used_car_depreciation_report.pdf
|-- results/
|   |-- benchmarks/              # benchmark model outputs
|   |-- error_analysis/          # segment-level error reports
|   `-- sample_scored/           # sample scoring outputs
|-- scripts/                     # data and modeling scripts
|-- src/used_car_project/        # reusable project utilities
|-- PORTFOLIO_SUMMARY.md
|-- LICENSE
|-- requirements.txt
`-- RUN_APP.bat
```

## Project Workflow

The project can be used at two levels:

### 1. Use the existing project outputs

You can inspect the included report assets, results, benchmark outputs, and run the web app directly using the bundled model and sample data.

### 2. Reproduce or extend the workflow

You can also place your own raw dataset in `data/external/` and rerun the pipeline:

1. Clean raw data
2. Generate EDA figures
3. Train or retrain a model
4. Benchmark alternative models
5. Score predictions
6. Analyze segment-level errors
7. Launch the web app

## Installation

Create a virtual environment and install dependencies:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How To Run The Web App

### Option A: Use the batch launcher

Double-click:

```text
RUN_APP.bat
```

### Option B: Run manually in the terminal

```bash
py -m streamlit run app/streamlit_app.py
```

After Streamlit starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## What The Web App Does

The Streamlit app provides:

- a project overview with dataset and model context
- an interactive valuation workbench
- estimated price for a selected vehicle profile
- depreciation curves by mileage and vehicle age
- nearest comparable listings from the sample data
- feature importance and benchmark model comparison

## Reproducible Commands

### Clean raw data

```bash
py scripts/clean_cars.py --input data/external/cars.csv --output data/processed/cars_clean.parquet
```

### Generate EDA report assets

```bash
py scripts/dataset_report.py --data data/processed/cars_clean.parquet --out reports/figures
```

### Train a LightGBM validation model

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

### Run segment-level error analysis

```bash
py scripts/error_analysis.py --scored results/sample_scored/scored_all.csv --out results/error_analysis
```

## Included Outputs

This upload-ready repository already includes:

- a sample cleaned dataset
- a trained model bundle
- headline metrics
- feature importance
- benchmark model results
- segment-level error analysis reports
- EDA figures
- a PDF report

## Current Results

### Prior full-run model evidence

- MAE: about `$1.7k`
- RMSE: about `$2.5k`
- R2: about `0.97`

### Sample benchmark results included in this repo

- Random Forest performed best on the included sample benchmark
- Linear Regression and Gradient Boosting were competitive but weaker
- Median Baseline performed much worse, confirming the task is learnable

## Files To Review First

- [PORTFOLIO_SUMMARY.md](C:/Users/Joyce/Desktop/study/used-car-depreciation/PORTFOLIO_SUMMARY.md)
- [docs/case_study.md](C:/Users/Joyce/Desktop/study/used-car-depreciation/docs/case_study.md)
- [app/streamlit_app.py](C:/Users/Joyce/Desktop/study/used-car-depreciation/app/streamlit_app.py)
- [results/benchmarks/benchmark_results.csv](C:/Users/Joyce/Desktop/study/used-car-depreciation/results/benchmarks/benchmark_results.csv)
- [reports/used_car_depreciation_report.pdf](C:/Users/Joyce/Desktop/study/used-car-depreciation/reports/used_car_depreciation_report.pdf)

## Notes For Other Users

- The repository is usable immediately with the included sample data and model.
- Large original raw files are not included because they are too large for a normal GitHub repository.
- If you want to rerun the full workflow on your own dataset, place your files in `data/external/` and follow the commands above.

## License

This project is released under the MIT License. See [LICENSE](C:/Users/Joyce/Desktop/study/used-car-depreciation/LICENSE).

# Used Car Depreciation

An end-to-end data project that analyzes how used vehicles lose value over time and builds a machine learning workflow to estimate resale price from listing-level attributes.

## Project Summary

Used-car pricing is noisy, inconsistent, and highly sensitive to mileage, age, and vehicle configuration. This project was built to answer a practical valuation question:

**Can we estimate a fair used-car price from listing attributes while also understanding the main drivers of depreciation?**

The final repository is organized as a reproducible and shareable data project rather than a single experimental notebook. It includes:

- data cleaning and preprocessing
- exploratory data analysis
- feature engineering
- predictive modeling
- benchmark comparison
- segment-level error analysis
- a Streamlit app for interactive valuation

## Problem Statement

In the used-car market, prices vary widely even within the same make and model. A useful pricing project should do more than output a prediction. It should also help explain:

- which variables matter most
- whether the model generalizes beyond a single split
- which vehicle segments are harder to predict
- how a predicted value compares with a listing's asking price

This project was designed with those goals in mind.

## Dataset and Data Preparation

The project uses used-vehicle listing data with fields such as:

- price
- year
- listed date
- mileage
- make and model
- trim
- body type
- fuel type
- transmission
- wheel system
- engine characteristics
- accident and fleet indicators

Because raw listing data is messy, the preparation step was a major part of the project. The cleaning workflow:

- standardized column names
- converted mixed-type numeric fields into usable numeric columns
- normalized boolean fields such as accident and fleet history
- converted engine displacement into a consistent liter-based feature
- derived vehicle age from listing date and model year
- handled missing and inconsistent categorical values
- created a project structure with raw, processed, sample, model, and results layers

The cleaning logic lives in [scripts/clean_cars.py](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/scripts/clean_cars.py).

## Exploratory Analysis

Before modeling, I used EDA to validate whether the data behaved in economically reasonable ways. The exploratory analysis focused on:

- price distribution across the market
- price vs. mileage
- price vs. vehicle age
- body type differences
- brand frequency
- monthly price trends

These plots helped confirm that depreciation patterns were present and also informed feature engineering choices. The generated figures are in [reports/figures](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/reports/figures).

## Feature Engineering

Several engineered features were added to make the model more useful on structured vehicle data:

- `age_years`
- `age_sq`
- `log_mileage`
- `engine_disp_l`
- `hp_per_l`
- standardized engine type labels
- normalized boolean indicators

These choices were motivated by the structure of the problem: depreciation is nonlinear, mileage effects are not strictly linear, and engine configuration carries pricing signal that raw text fields do not expose cleanly.

## Modeling Approach

The primary predictive model is a LightGBM regressor trained on tabular vehicle data.

LightGBM was chosen because:

- the data is structured and feature-rich
- price relationships are nonlinear
- categorical and numeric signals interact in complex ways
- gradient boosting is typically strong on tabular regression tasks

To keep the project honest, I did not stop at one model. I added benchmarking against simpler baselines so that performance could be interpreted in context rather than as a standalone number.

Included benchmark models:

- Median Baseline
- Linear Regression
- Gradient Boosting
- Random Forest

The benchmark script is [scripts/benchmark_models.py](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/scripts/benchmark_models.py).

## Validation Strategy

A pricing model can look strong if the split is too easy, so validation strategy matters.

When listing dates were available, I used a time-based split rather than relying only on a random split. This is closer to a realistic deployment setup because it tests how well the model predicts on newer listings using older ones for training.

This design choice matters because it reduces the risk of overestimating model quality through overly optimistic sampling.

## Results

### Historical full-run model performance

- MAE: about **$1.7k**
- RMSE: about **$2.5k**
- R²: about **0.97**

### Included sample benchmark result

The benchmark results included in this repository show:

- **Random Forest** performed best on the sample benchmark
- **Linear Regression** remained a strong and interpretable baseline
- **Gradient Boosting** was competitive
- **Median Baseline** performed substantially worse

Files:

- [results/metrics.json](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/metrics.json)
- [results/benchmarks/benchmark_results.csv](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/benchmarks/benchmark_results.csv)

## Error Analysis

Strong overall metrics do not guarantee consistent behavior across all market segments. To make the project more realistic and useful, I added segment-level error analysis.

The project evaluates error by:

- make
- body type
- price band
- age band
- mileage band

This makes it easier to identify where the model is more reliable and where it may need more data or better features.

Files:

- [results/error_analysis](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/results/error_analysis)

## Streamlit Application

The repository includes a Streamlit app that turns the project into an interactive tool rather than a static analysis artifact.

The app supports:

- project overview and context
- interactive vehicle input
- model-estimated resale price
- comparison between estimated price and asking price
- depreciation curves by mileage and age
- comparable sample listings
- model insight and benchmark summaries

App file:

- [app/streamlit_app.py](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/app/streamlit_app.py)

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
|-- scripts/                     # data and modeling scripts
|-- src/used_car_project/        # reusable project utilities
|-- INTERVIEW_GUIDE.md
|-- PORTFOLIO_SUMMARY.md
|-- RUN_APP.bat
|-- requirements.txt
`-- README.md
```

## How To Run The Project

### 1. Install dependencies

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the web app

#### Option A

Double-click:

```text
RUN_APP.bat
```

#### Option B

Run manually:

```bash
py -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Reproducing the Workflow

### Clean raw data

```bash
py scripts/clean_cars.py --input data/external/cars.csv --output data/processed/cars_clean.parquet
```

### Generate EDA assets

```bash
py scripts/dataset_report.py --data data/processed/cars_clean.parquet --out reports/figures
```

### Train the main validation model

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

## Key Takeaways

- Data cleaning quality mattered as much as model choice.
- Age and mileage were the dominant depreciation drivers.
- Vehicle configuration added meaningful secondary signal.
- Benchmarking and grouped error analysis made the project more credible.
- Turning the work into an app made the project easier to demonstrate and reuse.

## Limitations

- The repository includes sample data and lightweight artifacts rather than the full original raw dataset.
- Performance can vary across market segments, especially where data is sparse.
- Location effects and some market dynamics are not fully modeled in the current version.

## Future Improvements

- add SHAP-based interpretability
- improve comparable-listing logic in the app
- retrain on a larger curated processed dataset inside the final structure
- extend validation with richer temporal or segment-based evaluation
- add uncertainty estimates to predictions

## Additional Project Notes

For a shorter project pitch or interview-oriented summary, see:

- [PORTFOLIO_SUMMARY.md](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/PORTFOLIO_SUMMARY.md)
- [INTERVIEW_GUIDE.md](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/INTERVIEW_GUIDE.md)

## License

This project is released under the MIT License. See [LICENSE](C:/Users/Joyce/Desktop/study/used-car-depreciation-github-ready/LICENSE).

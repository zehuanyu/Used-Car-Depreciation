# Used Car Depreciation Case Study

## Business Question

How strongly do age, mileage, and vehicle attributes drive used-car depreciation, and how accurately can we estimate a fair market price from listing-level features?

## Why This Project Matters

- Buyers want a quick estimate of whether a listing is fairly priced.
- Sellers want to understand how much value a vehicle has already lost.
- Analysts want interpretable drivers of depreciation rather than only a black-box prediction.

## Analytical Workflow

1. Clean raw listing data and standardize vehicle attributes.
2. Explore how price changes across age, mileage, body type, and brand.
3. Engineer features that capture depreciation patterns.
4. Train and compare models, not just one final model.
5. Analyze where the model performs well and where it breaks down.
6. Present both predictive output and business-facing insights.

## Current Findings

- Age and mileage are the strongest predictors of resale value.
- Engine configuration and trim add meaningful lift after age and mileage.
- The best current model achieved strong accuracy on the original full evaluation:
  - MAE: about $1.7k
  - RMSE: about $2.5k
  - R2: about 0.97
- Performance still needs closer inspection by segment, especially for rarer brands and body types.

## Upgrade Goals

- Add model benchmarking against simpler baselines.
- Add segment-level error analysis.
- Improve the app so it explains depreciation behavior, not only price prediction.
- Make the repository easier to review, rerun, and present as a portfolio project.

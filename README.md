## Mapping subnational gender gaps in internet and mobile adoption using social media data

This repository contains code and materials to replicate ["Mapping subnational gender gaps in internet and mobile adoption using social media data"]([https://osf.io/](https://osf.io/preprints/socarxiv/qnzsw_v2))

### Replication Package

This repository includes code to replicate all figures and tables in the paper. Please note that to run the bootstrap replication code, you will need access to all DHS surveys from 2015-2022.  

1. Clone this repository
2. Download all data from the `/data/` repository
2. Run the `00_run_all.Rmd` script, which will run all code (or run scripts `01` to `10` individually)

#### Computing 

All computations were carried out on 2023 MacBook Pro with an Apple M2 Pro chip, 16GB memory, and Sonoma 14.1 operating system.

#### Data 

Please download the open science framework repository. In the same directory as your /code/ folder, create a /data/ folder and place all data files in this folder.

Note: Running script 12, which create a table with number of interviews per DHS survey, requires access to the underlying women and men's DHS microdata recodes. These files are available upon registration from the [DHS program](https://dhsprogram.com/)

#### Code 

After downloading the required data, researchers can run the following script to replicate all figures and tables: 
  
- `00_run_all.Rmd` - this file runs all scripts. 

Alternatively, researchers can run the following files individually in order: 
  
- `00_run_all.Rmd` – Master script to run the full analysis pipeline in sequence.
- `01_fit_models_crossvalidation.Rmd` – Fits subnational prediction models using both leave-one-country-out (LOCO) and 10-fold cross-validation.
- `02_make_predictions.Rmd` – Uses the trained models to generate predictions for all LMICs and estimates associated uncertainty.
- `03_model_performance.Rmd` – Produces summary figures and tables for overall model performance metrics (e.g., RMSE, R²).
- `04_within_country_analysis.Rmd` – Evaluates and visualizes within-country performance across models.
- `05_geo_map.Rmd` – Generates geographic visualizations (maps) of predicted values and associated prediction errors.
- `06_feature_importance.Rmd` – Assesses and visualizes the relative importance of input features for each model.
- `07_superlearner_weights.Rmd` – Computes and reports SuperLearner ensemble weights, indicating the relative contribution of each learner.
- `08_residual_analysis.Rmd` – Analyzes and plots model residuals to assess spatial and structural model fit.
- `09_var_explained.Rmd` – Decomposes variance explained into within- and between-country components.
- `10_bootstrap_analysis.Rmd` – Runs bootstrap procedures to estimate uncertainty around DHS ground truth.
- `11_disparities_plots.Rmd` – Generates plots to explore subnational disparities and patterns in predicted outcomes.
- `12_dhs_table.Rmd` – Constructs a table summarizing the number of individuals sampled in each DHS survey (requires access to raw DHS microdata).
- `13_var_coverage_table.Rmd` – Produces a table indicating availability of key predictor variables across country-year units.
- `14_trends.Rmd` – Analyzes and visualizes time trends in model predictions and covariates.
- `15_tanzania_analysis.Rmd` – Focused analysis of model results and validation in Tanzania as a case study.
- `16_validation_of_uncertainty.Rmd` – Validates uncertainty estimates through empirical coverage and interval accuracy checks.
- `17_lsms_comparison.Rmd` – Compares subnational predictions against LSMS. 
- `18_mics_comparison.Rmd` – Compares subnational predictions against MICS
- `19_mics_comparison_national.Rmd` – Compares predictions to MICS at the national level for robustness.
- `20_trends_validation.Rmd` – Validates predicted trends over time against external benchmarks and survey data.
- `21_trend_validation_aggregate.Rmd` – Aggregates trend validations across all countries and produces summary figures.

### Authors

- [Casey Breen](https:://caseybreen.com)
- Masoomali Fatehkia
- Jiani Yan
- Xinyi Zhao
- Douglas R. Leasure
- Ingmar Weber
- Ridhi Kashyap



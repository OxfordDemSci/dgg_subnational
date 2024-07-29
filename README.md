## Mapping subnational gender gaps in internet and mobile adoption using social media data

This repository contains code and materials to replicate ["Mapping subnational gender gaps in internet and mobile adoption using social media data"](https://osf.io/)

### Replication Package

This repository includes code to replicate all figures and tables in the paper. Please note that to run the bootstrap replication code, you will need access to all DHS surveys from 2015-2022.  

1. Clone this repository
2. Download all data from the `/data/` repository
2. Run the `00_run_all.Rmd` script, which will run all code (or run scripts `01` to `10` individually)

#### Computing 

All computations were carried out on 2023 MacBook Pro with an Apple M2 Pro chip, 16GB memory, and Sonoma 14.1 operating system.

#### Data 

Please download the open science framework repository. In the same directory as your /code/ folder, create a /data/ folder and place all data files in this folder.

#### Code 

After downloading the required data, researchers can run the following script to replicate all figures and tables: 
  
  - `00_run_all.Rmd` - this file runs all scripts. 

Alternatively, researchers can run the following files individually in order: 
  
- `01_fit_models_crossvalidation.Rmd` - Fit subnational models using both leave-one-country-out (LOCO) and 10-fold cross-validation.
- `02_make_predictions.Rmd` - Make predictions for all LMICs using full training data and estimate uncertainty 
- `03_model_performance.Rmd` - Generate figures and tables summarizing model performance 
- `04_within_country_analysis.Rmd` - Generate figures and tables summarizing within country performance 
- `05_geo_map.Rmd` - Geographic visualizations of model predictions and error
- `06_feature_importance.Rmd` - Analysis of most important features in a model 
- `07_superlearner_weights.Rmd` - Run all (full) models and get weights from Superlearner showing relative contribution of each learner 
- `08_residual_analysis.Rmd` - Calculate and plot model residuals 
- `09_var_explained.Rmd` - Analysis of within vs. between country variance 
- `10_bootstrap_analysis.Rmd` - Bootstrap analysis to estimate the uncertainty in the R-squared values


### Authors

- [Casey Breen](caseybreen.com)
- Masoomali Fatehkia
- Jiani Yan
- Xinyi Zhao
- Douglas R. Leasure
- Ingmar Weber
- Ridhi Kashyap



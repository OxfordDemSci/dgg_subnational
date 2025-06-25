## Mapping subnational gender gaps in internet and mobile adoption using social media data

[![OSF](https://img.shields.io/badge/OSF-project-blue)](https://osf.io/5e8wf/)
[![Generic badge](https://img.shields.io/badge/R-4.3.1-orange.svg)](https://cran.r-project.org/bin/macosx/)
[![Generic badge](https://img.shields.io/badge/License-GNU-<green>.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Generic badge](https://img.shields.io/badge/Maintained-Yes-red.svg)]()


This repository contains code and materials to replicate the paper ["Mapping subnational gender gaps in internet and mobile adoption using social media data"]([https://osf.io/](https://osf.io/preprints/socarxiv/qnzsw_v2))


**Abstract**: The digital revolution has ushered in many societal and economic benefits. Yet access to
digital technologies such as mobile phones and internet remains highly unequal, especially
by gender in the context of low- and middle-income countries (LMICs). While national-level
estimates are increasingly available for many countries, reliable, quantitative estimates of
digital gender inequalities at the subnational level are lacking. These estimates, however, are
essential for monitoring gaps within countries and implementing targeted interventions within
the global sustainable development goals, which emphasize the need to close inequalities
both between and within countries. We develop estimates of internet and mobile adoption by
gender and digital gender gaps at the subnational level for 2,075 regions in 117 LMICs from
2015 through 2025, a context where digital penetration is low and national-level gender gaps
disfavoring women are large. We construct these estimates by applying machine-learning
algorithms to Facebook user counts, geospatial data, development indicators, and population
composition data. We calibrate and assess the performance of these algorithms using ground-
truth data from subnationally-representative household survey data from 33 LMICs. Our
results reveal striking disparities in access to mobile and internet technologies between and
within LMICs. These disparities imply that as of 2025, women are 19% less likely to use the
internet and 8% less likely to own a mobile phone in LMICs, corresponding to over 190 million
fewer women owning a mobile phone and over 320 million fewer women using the internet. 

### Interactive Dashboard

------------

An interactive dashboard displaying these estimates is available here: [digitalgendergaps.org/dashboard/](https://digitalgendergaps.org/dashboard/)


### Replication Package

------------


This repository includes code to replicate all figures and tables in the paper. To replicate, follow the steps outlined below: 

1. Clone this repository
2. Download the `data.zip` file from the [OSF repository](https://osf.io/5e8wf/), move it to the root level of this repository, and unzip it. 
2. Run the `00_run_all.Rmd` script, which will run all code (or run scripts `01` to `21` individually)

### Data 

------------


Please download the `data.zip` file from the Open Science Framework repository. At the root level of this repository (alongside the `code/` folder), unzip 
the archive to create a `data/` folder containing all required data files for the analysis.

**Note**: Running the scripts `10_bootstrap_analysis.Rmd` and `12_dhs_table.Rmd` requires access to the underlying women
and men's DHS microdata files. These files are available from the [DHS program](https://dhsprogram.com/) following registration. 

### Code 

------------


After downloading the required data, researchers can run the following script to replicate all figures and tables in the paper: 
  
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

- [Casey Breen](https://caseybreen.com)
- Masoomali Fatehkia
- Jiani Yan
- Xinyi Zhao
- Douglas R. Leasure
- Ingmar Weber
- [Ridhi Kashyap](https://www.nuffield.ox.ac.uk/people/profiles/ridhi-kashyap/)


## Replication

All analyses and computations were carried out on 2023 MacBook Pro with an Apple M2 Pro chip, 16GB memory, and Sonoma 14.1 operating system.

All analyses were originally conducted using R version 4.3.1 and the package versions recorded in the attached session info at the bottom of the README. 
Re-running the pipeline with updated R or package versions, or a different seed, may yield minor numerical differences. 
These do not affect the paper’s results or conclusions.

**Note**: The SL3 package is no longer available on CRAN, and needs to be installed directly from [Github](https://github.com/tlverse/sl3):  

```
remotes::install_github("tlverse/sl3")
```


## Acknowledgements 

For helpful discussions and feedback, we thank Dennis Feehan, Michelle Niemann, Anna Bolgrien,
Sharada Srinivasan, and participants of the PAA 2024 “Digital Demography” session, the EPC 2024 “Modelling Subnational and
Spatial Variation” session, the Digital Inequality and Sustainable Development workshop, the Johns Hopkins Population Health
seminar series, the Max Planck Institute for Demographic Research Digital and Computational Demography seminar series, the
Sciences Po Digital Inequality Symposium, and the Leverhulme Centre for Demographic Science seminar series. We thank Jiaxuan
Li for excellent research assistance. The authors acknowledge financial support from the Bill and Melinda Gates Foundation
(INV-045370), the Leverhulme Trust (Grant RC-2018-003) for the Leverhulme Centre for Demographic Science, and the Alexander
von Humboldt Foundation and its founder, the Federal Ministry of Education and Research (Bundesministerium fur Bildung und
Forschung).


## Session info:

```
sessionInfo()

R version 4.3.1 (2023-06-16)
Platform: aarch64-apple-darwin20 (64-bit)
Running under: macOS 15.4.1

Matrix products: default
BLAS:   /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib 
LAPACK: /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.11.0

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

time zone: Europe/London
tzcode source: internal

attached base packages:
[1] parallel  splines   stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] hal9001_0.4.3         Rcpp_1.0.13-1         xgboost_1.7.5.1       gbm_2.1.8.1           polspline_1.1.23      glmnet_4.1-7          Matrix_1.6-0          ggalt_0.4.0          
 [9] ineq_0.2-13           conflicted_1.2.0      lubridate_1.9.3       forcats_1.0.0         stringr_1.5.1         dplyr_1.1.4           purrr_1.0.2           readr_2.1.4          
[17] tidyr_1.3.1           tibble_3.2.1          tidyverse_2.0.0       haven_2.5.3           xtable_1.8-4          countrycode_1.5.0     ggspatial_1.1.8       viridis_0.6.4        
[25] viridisLite_0.4.2     colorspace_2.1-1      tigris_2.0.3          patchwork_1.3.0       terra_1.7-39          raster_3.6-23         sp_2.0-0              stars_0.6-3          
[33] sf_1.0-16             abind_1.4-8           origami_1.0.7         ranger_0.15.1         ggpubr_0.6.0          cowplot_1.1.1         caret_6.0-94          lattice_0.21-8       
[41] ggplot2_3.5.1         doParallel_1.0.17     iterators_1.0.14      sl3_1.4.4             SuperLearner_2.0-28.1 gam_1.22-2            foreach_1.5.2         nnls_1.5             
[49] knitr_1.43            data.table_1.16.4     pacman_0.5.1          here_1.0.1           

loaded via a namespace (and not attached):
  [1] R.oo_1.25.0          janitor_2.2.0        hardhat_1.3.0        pROC_1.18.5          rpart_4.1.19         lifecycle_1.0.4      Rdpack_2.6.1         rstatix_0.7.2       
  [9] rprojroot_2.0.3      vroom_1.6.3          globals_0.16.2       MASS_7.3-60          backports_1.5.0      magrittr_2.0.3       rmarkdown_2.23       yaml_2.3.7          
 [17] DBI_1.2.3            RColorBrewer_1.1-3   maps_3.4.1           R.utils_2.12.2       nnet_7.3-19          rappdirs_0.3.3       ipred_0.9-14         lava_1.7.2.1        
 [25] listenv_0.9.0        units_0.8-3          parallelly_1.36.0    codetools_0.2-19     tidyselect_1.2.1     shape_1.4.6          imputeMissings_0.0.3 farver_2.1.2        
 [33] ash_1.0-15           stats4_4.3.1         jsonlite_1.8.9       e1071_1.7-13         survival_3.5-5       tools_4.3.1          progress_1.2.2       glue_1.8.0          
 [41] prodlim_2023.03.31   gridExtra_2.3        Rttf2pt1_1.3.12      xfun_0.39            withr_3.0.2          fastmap_1.1.1        fansi_1.0.6          digest_0.6.33       
 [49] timechange_0.3.0     R6_2.5.1             R.methodsS3_1.8.2    utf8_1.2.4           generics_0.1.3       recipes_1.0.10       class_7.3-22         prettyunits_1.1.1   
 [57] httr_1.4.7           htmlwidgets_1.6.2    ModelMetrics_1.2.2.2 pkgconfig_2.0.3      gtable_0.3.6         timeDate_4022.108    htmltools_0.5.5      carData_3.0-5       
 [65] scales_1.3.0         delayed_0.4.0        snakecase_0.11.0     gower_1.0.1          rstudioapi_0.15.0    tzdb_0.4.0           reshape2_1.4.4       uuid_1.1-0          
 [73] rstackdeque_1.1.1    visNetwork_2.1.2     checkmate_2.3.2      nlme_3.1-162         proxy_0.4-27         cachem_1.0.8         KernSmooth_2.23-21   extrafont_0.19      
 [81] pillar_1.9.0         grid_4.3.1           vctrs_0.6.5          randomForest_4.7-1.1 car_3.1-2            extrafontdb_1.0      evaluate_0.21        BBmisc_1.13         
 [89] cli_3.6.3            compiler_4.3.1       rlang_1.1.4          crayon_1.5.2         future.apply_1.11.0  ggsignif_0.6.4       classInt_0.4-9       plyr_1.8.8          
 [97] stringi_1.8.4        assertthat_0.2.1     munsell_0.5.1        proj4_1.0-14         hms_1.1.3            bit64_4.0.5          future_1.33.0        rbibutils_2.3       
[105] ROCR_1.0-11          igraph_1.5.0         broom_1.0.5          memoise_2.0.1        lwgeom_0.2-13        bit_4.0.5           
```


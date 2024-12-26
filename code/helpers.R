########################################
# Helper Functions for Subnational Modeling
# Author: Casey Breen
#
########################################

# Install and load pacman if not already installed
if (!require("pacman")) install.packages("pacman")

# Use pacman to load all necessary packages
pacman::p_load(
  here,         # relative file paths
  data.table,   # data wrangling
  knitr,        # dynamic report generation
  SuperLearner, # ensemble learning
  sl3,          # machine learning
  doParallel,   # parallel computing
  caret,        # machine learning 
  cowplot,      # pretty plots
  ggpubr,       # publication-ready plots
  ranger,       # random forests
  origami,      # cross-validation
  nnls,         # non-negative least squares
  stars,        # spatiotemporal data handling
  raster,       # raster data handling
  terra,        # raster data handling
  sf,           # vector data handling
  patchwork,    # arranging figures
  tigris,       # county border data
  colorspace,   # color scales
  viridis,      # color scales for plots
  ggspatial,    # north arrow and scale bar
  countrycode,  # country code 
  xtable,       # package for making latex tables
  haven,        # read stata files
  tidyverse,    # data manipulation and visualization
  conflicted,    # custom functions
  caret,
  ineq # machine learning 
)

## custom functions 
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

## Source .Rmd 

source_rmd = function(file, ...) {
  tmp_file = tempfile(fileext=".R")
  on.exit(unlink(tmp_file), add = TRUE)
  knitr::purl(file, output=tmp_file, quiet = T)
  source(file = tmp_file, ...)
}

## execute source 
execute_source <- function(file) {
  tryCatch(
    source(file),
    error = function(e) {
      cat("Error:", e$message, "\n")
    }
  )
}



## custom color schemes 
## custom color schemes
cudb <- c("#49b7fc", "#ff7b00", "#17d898", "#ff0083", "#0015ff", "#e5d200", "#999999")
cud <- c("#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#E69F00", "#F0E442", "#999999")
cbp1 <- c("#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

## white background instead of transparent
ggsave <- function(filename, plot = last_plot(), ..., bg = "white") {
  ggplot2::ggsave(filename, plot, ..., bg = bg)
}


# Compute R^2 functions
compute_r2 <- function(predicted, observed) {
  ss_res <- sum((observed - predicted)^2)
  ss_tot <- sum((observed - mean(observed))^2)
  r2 <- 1 - ss_res / ss_tot
  return(r2)
}

## compute rmse
compute_rmse <- function(predicted, observed) {
  mse <- sum((observed - predicted)^2) / length(observed)
  rmse <- sqrt(mse)
  return(rmse)
}

## compute mape
compute_mape <- function(predicted, observed) {
  mape <- 100 * mean(abs((observed - predicted) / observed))
  return(mape)
}


## compute mare
compute_mare <- function(predicted, observed) {
  mare <- mean(abs((observed - predicted) / observed))
  return(mare)
}

## compute mae
compute_mae <- function(predicted, observed) {
  mae <- mean(abs(observed - predicted))
  return(mae)
}


## run "leave-one-country out" model

# This function, run_model, takes three arguments:
#
# 1. gadm1: the dataset.
# 2. outcome: the outcome variable.
# 3. covar_list: a list of vectors containing covariate column names. It defaults to list(fb_feats, off_feats).
# The function returns a dataframe in the long format (predictions_df_long) with model predictions for the hold-out

# Example usage:
# predictions <- run_model(gadm1, "perc_used_internet_past12months_wght_age_15_to_49_wom", list(fb_feats, off_feats))


run_loco_model <- function(data, outcome, covar_list = c(fb_feats, off_feats)) {
  # Define list of countries
  countries <- unique(data$country)
  
  # Initialize counter and list to store prediction results
  i <- 1
  predictions_list <- list()
  
  # Loop through each country to perform a leave-one-out (LOO) validation
  for (holdout_country in countries) {
    # Split the data into training and holdout sets based on country
    gadm1_train <- data %>% filter(country != holdout_country)
    gadm1_holdout <- data %>% filter(country == holdout_country)
    
    # Define learners for the super learner ensemble
    lrn_glm <- make_learner(Lrnr_glm)
    lrn_lasso <- make_learner(Lrnr_glmnet)
    lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
    lrn_enet.5 <- make_learner(Lrnr_glmnet, alpha = 0.5)
    lrn_polspline <- Lrnr_polspline$new()
    lrn_ranger100 <- make_learner(Lrnr_ranger, num.trees = 100)
    lrn_hal_faster <- Lrnr_hal9001$new(max_degree = 2, reduce_basis = 0.05)
    lrnr_gbm <- make_learner(Lrnr_gbm)
    lrnr_xgb <- Lrnr_xgboost$new()
    
    # Create list of learners and assign names
    learners <- c(lrn_glm, lrn_lasso, lrn_ridge, lrn_enet.5, lrn_polspline, lrn_ranger100, lrnr_gbm, lrnr_xgb)
    names(learners) <- c("glm", "lasso", "ridge", "elastic_new", "poly_spline", "ranger", "gbm", "xgb")
    
    # Create super learner stack
    stack <- make_learner(Stack, learners)
    
    # Extract covariates from training data
    covars <- colnames(gadm1_train %>% dplyr::select(all_of(covar_list)))
    
    # Create k-fold CV with countries as clusters
    folds <- make_folds(cluster_ids = gadm1_train$country)
    
    # Define SL3 task for training data
    superlearner_task <- make_sl3_Task(
      data = gadm1_train,
      folds = folds,
      covariates = covars,
      outcome = outcome,
      outcome_type = "continuous",
      drop_missing_outcome = TRUE
    )
    
    # Create super learner object
    sl <- make_learner(Lrnr_sl, learners = stack, metalearner = Lrnr_nnls$new())
    
    # Train the super learner model on the task
    sl_fit <- sl$train(task = superlearner_task)
    
    # Create SL3 task for holdout data for making predictions
    prediction_task <- make_sl3_Task(
      data = gadm1_holdout,
      covariates = covars,
      outcome = outcome,
      outcome_type = "continuous"
    )
    
    # Make predictions using various learners
    prediction_df <- data.frame(
      glm = sl_fit$learner_fits$glm$predict(task = prediction_task),
      xgboost = sl_fit$learner_fits$xgb$predict(task = prediction_task),
      ranger = sl_fit$learner_fits$ranger$predict(task = prediction_task),
      gbm = sl_fit$learner_fits$gbm$predict(task = prediction_task),
      lasso = sl_fit$learner_fits$lasso$predict(task = prediction_task),
      ridge = sl_fit$learner_fits$ridge$predict(task = prediction_task),
      superlearner = sl_fit$predict(task = prediction_task),
      observed = gadm1_holdout %>% pull(!!sym(outcome)),
      gid_1 = gadm1_holdout %>% pull(gid_1),
      country = holdout_country
    )
    
    # Store predictions in list
    predictions_list[[i]] <- prediction_df
    
    # Print out the current holdout country
    # cat(holdout_country, "\n")
    
    i <- i + 1
  }
  
  # Combine all prediction results into a single dataframe
  predictions_df <- bind_rows(predictions_list)
  
  covar_list_filtered <- covar_list[!covar_list %in% "dhsyear"]
  
  # Determine feature set based on covar_list
  feature_set <- ifelse(any(covar_list_filtered %in% fb_feats) && any(covar_list_filtered %in% off_feats),
                        "fb_and_offline",
                        ifelse(all(covar_list_filtered %in% fb_feats), "fb", "offline")
  )
  
  # Reshape dataframe to long format and add feature set column
  predictions_df_long <- predictions_df %>%
    pivot_longer(-c(country, observed, gid_1), names_to = "model", values_to = "predicted") %>%
    mutate(outcome = outcome, feature_set = feature_set)
  
  return(predictions_df_long)
}

## run model 10fold
run_model_10fold <- function(data, outcome, covar_list = c(fb_feats, off_feats)) {
  # Initialize list to store prediction results
  predictions_list <- list()
  
  # Define learners for the super learner ensemble
  lrn_glm <- make_learner(Lrnr_glm)
  lrn_lasso <- make_learner(Lrnr_glmnet)
  lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
  lrn_enet.5 <- make_learner(Lrnr_glmnet, alpha = 0.5)
  lrn_polspline <- Lrnr_polspline$new()
  lrn_ranger100 <- make_learner(Lrnr_ranger, num.trees = 100)
  lrnr_gbm <- make_learner(Lrnr_gbm)
  lrnr_xgb <- Lrnr_xgboost$new()
  
  # Create list of learners and assign names
  learners <- c(lrn_glm, lrn_lasso, lrn_ridge, lrn_enet.5, lrn_polspline, lrn_ranger100, lrnr_gbm, lrnr_xgb)
  names(learners) <- c("glm", "lasso", "ridge", "elastic_new", "poly_spline", "ranger", "gbm", "xgb")
  
  # Create super learner stack
  stack <- make_learner(Stack, learners)
  
  # Extract covariates from data
  covars <- colnames(data %>% dplyr::select(all_of(covar_list)))
  
  # Create 10-fold CV
  folds <- make_folds(data, V = 10)
  
  # Loop over folds
  for (i in 1:10) {
    train_indices <- folds[[i]][["training_set"]]
    test_indices <- folds[[i]][["validation_set"]]
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    ##
    inner_folds <- make_folds(train_data, V = 10)
    
    # Define SL3 task for training data
    superlearner_task <- make_sl3_Task(
      data = as.data.frame(train_data),
      folds = inner_folds,
      covariates = covars,
      outcome = outcome,
      outcome_type = "continuous",
      drop_missing_outcome = TRUE
    )
    
    # Create super learner object
    sl <- make_learner(Lrnr_sl, learners = stack, metalearner = Lrnr_nnls$new())
    
    # Train the super learner model on the task
    sl_fit <- sl$train(task = superlearner_task)
    
    # Create SL3 task for holdout data for making predictions
    prediction_task <- make_sl3_Task(
      data = test_data,
      covariates = covars,
      outcome = outcome,
      outcome_type = "continuous"
    )
    
    # Make predictions using various learners
    prediction_df <- data.frame(
      glm = sl_fit$learner_fits$glm$predict(task = prediction_task),
      xgboost = sl_fit$learner_fits$xgb$predict(task = prediction_task),
      ranger = sl_fit$learner_fits$ranger$predict(task = prediction_task),
      gbm = sl_fit$learner_fits$gbm$predict(task = prediction_task),
      lasso = sl_fit$learner_fits$lasso$predict(task = prediction_task),
      ridge = sl_fit$learner_fits$ridge$predict(task = prediction_task),
      superlearner = sl_fit$predict(task = prediction_task),
      observed = test_data %>% pull(outcome),
      country = test_data %>% pull(country),
      gid_1 = test_data %>% pull(gid_1)
    )
    
    # Store predictions in list
    predictions_list[[i]] <- prediction_df
  }
  
  # Combine all prediction results into a single dataframe
  predictions_df <- bind_rows(predictions_list)
  
  covar_list_filtered <- covar_list[!covar_list %in% "dhsyear"]
  
  # Determine feature set based on covar_list
  feature_set <- ifelse(any(covar_list_filtered %in% fb_feats) && any(covar_list_filtered %in% off_feats),
                        "fb_and_offline",
                        ifelse(all(covar_list_filtered %in% fb_feats), "fb", "offline")
  )
  
  # Reshape dataframe to long format and add feature set column
  predictions_df_long <- predictions_df %>%
    pivot_longer(-c(observed, country, gid_1), names_to = "model", values_to = "predicted") %>%
    mutate(outcome = outcome, feature_set = feature_set)
  
  return(predictions_df_long)
}


## fit simple superlearner model and make predictions
superlearner_train_and_predict <- function(data,
                                           outcome,
                                           covar_list = c(fb_feats, off_feats),
                                           predict_data = NULL,
                                           return_fit = NULL) {
  
  
  # Define learners for the super learner ensemble
  lrn_glm <- make_learner(Lrnr_glm)
  lrn_lasso <- make_learner(Lrnr_glmnet)
  lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
  lrn_enet.5 <- make_learner(Lrnr_glmnet, alpha = 0.5)
  lrn_polspline <- Lrnr_polspline$new()
  lrn_ranger100 <- make_learner(Lrnr_ranger, num.trees = 100)
  lrn_hal_faster <- Lrnr_hal9001$new(max_degree = 2, reduce_basis = 0.05)
  lrnr_gbm <- make_learner(Lrnr_gbm)
  lrnr_xgb <- Lrnr_xgboost$new()
  
  # Create list of learners and assign names
  learners <- c(lrn_glm, lrn_lasso, lrn_ridge, lrn_enet.5, lrn_polspline, lrn_ranger100, lrnr_gbm, lrnr_xgb)
  names(learners) <- c("glm", "lasso", "ridge", "elastic_new", "poly_spline", "ranger", "gbm", "xgb")
  
  # Create super learner stack
  stack <- make_learner(Stack, learners)
  
  # Extract covariates from training data
  covars <- colnames(data %>% dplyr::select(all_of(covar_list)))
  
  
  # Create k-fold CV with countries as clusters
  folds <- make_folds(cluster_ids = data$country)
  
  # Define SL3 task for training data
  superlearner_task <- make_sl3_Task(
    data = data,
    folds = folds,
    covariates = covars,
    outcome = outcome,
    outcome_type = "continuous",
    drop_missing_outcome = TRUE
  )
  
  # Create super learner object
  sl <- make_learner(Lrnr_sl, learners = stack, metalearner = Lrnr_nnls$new())
  
  # Train the super learner model on the task
  sl_fit <- sl$train(task = superlearner_task)
  
  # Create SL3 task for secondary data for making predictions
  prediction_task <- make_sl3_Task(
    data = predict_data,
    covariates = covars,
    outcome = outcome,
    outcome_type = "continuous"
  )
  
  # Make predictions using superlearner
  secondary_predictions <- data.frame(
    superlearner = sl_fit$predict(task = prediction_task),
    year = predict_data %>% pull(year),
    month = predict_data %>% pull(month),
    observed = predict_data %>% pull(outcome),
    country = predict_data %>% pull(country),
    gid_1 = predict_data %>% pull(gid_1),
    outcome = outcome
  )
  
  if (!is.null(return_fit)) {
    return(list(sl_fit = sl_fit, secondary_predictions = secondary_predictions))
  } else {
    return(secondary_predictions)
  }
}

## ---- Variables ----
dhs_vars <- tolower(c(
  "perc_ever_used_internet_wght_age_15_to_49_wom",
  "perc_used_internet_past12months_wght_age_15_to_49_wom",
  "perc_owns_mobile_telephone_wght_age_15_to_49_wom",
  "perc_ever_used_internet_wght_age_15_to_49_men",
  "perc_used_internet_past12months_wght_age_15_to_49_men",
  "perc_owns_mobile_telephone_wght_age_15_to_49_men",
  "perc_ever_used_internet_wght_age_15_to_49_fm_ratio",
  "perc_used_internet_past12months_wght_age_15_to_49_fm_ratio",
  "perc_owns_mobile_telephone_wght_age_15_to_49_fm_ratio"
))

fb_feats <- c(
  "fb_pntr_18p_male",
  "fb_pntr_18p_female",
  "all_devices_age_18_plus_gg",
  "fb_pntr_18p_female_national",
  "fb_pntr_18p_male_national",
  "all_devices_age_18_plus_gg_national",
  "ios_age_13_plus_female_frac_2024",
  "ios_age_13_plus_male_frac_2024",
  'wifi_age_13_plus_female_frac_2024',
  'wifi_age_13_plus_male_frac_2024', 
  'x4g_age_13_plus_female_frac_2024',
  'x4g_age_13_plus_male_frac_2024'
)

# fb_feats <- c(
#   "fb_pntr_13p_male_dhsyear_impute",
#   "fb_pntr_13p_female_dhsyear_impute",
#   "all_devices_age_13_plus_gg_dhsyear_impute",
#   "ios_age_13_plus_female_frac_2024",
#   "ios_age_13_plus_male_frac_2024",
#   "wifi_age_13_plus_female_frac_2024",
#   "wifi_age_13_plus_male_frac_2024",
#   "x4g_age_13_plus_female_frac_2024",
#   "x4g_age_13_plus_male_frac_2024",
#   "fb_pntr_13p_male_dhsyear_national",
#   "fb_pntr_13p_female_dhsyear_national",
#   "all_devices_age_13_plus_gg_dhsyear_national"
# )


off_feats <- tolower(c(
  "nl_mean_zscore",
  # "rwi_mean",
  # "rwi_mean_wght",
  "pop_density_zscore",
  "dhsyear",
  # "pop_all_age_female_zscore",
  # "pop_all_age_male_zscore",
  "subnational_gdi",
  "subnational_hdi_males",
  "subnational_hdi_females",
  "educational_index_females",
  "educational_index_males",
  "income_index_females",
  "income_index_males",
  #  "internet_speed",
  "continent",
  "hdi",
  "gdi",
  "rel_date",
  "rel_date_squared"
))



perform_cross_validation_loco <- function(data, dependent_var, features, country_col) {
  results_df <- data.frame(feature = character(), avg_r_squared = numeric(), stringsAsFactors = FALSE)
  countries <- unique(data[[country_col]])
  
  for (feat in features) {
    r_squared_values <- numeric(length(countries))
    
    for (country_holdout in countries) {
      # Split the data into training and testing based on the country
      testData <- data[data[[country_col]] == country_holdout, ]
      trainData <- data[data[[country_col]] != country_holdout, ]
      
      # Construct the formula
      formula_str <- as.formula(paste(dependent_var, "~", feat))
      
      # Fit the model on training data
      model <- lm(formula = formula_str, data = trainData)
      
      # Predict on testing data
      predictions <- predict(model, newdata = testData)
      
      # Calculate R^2
      ss_tot <- sum((testData[[dependent_var]] - mean(trainData[[dependent_var]]))^2)
      ss_res <- sum((testData[[dependent_var]] - predictions)^2)
      r_squared_values[which(countries == country_holdout)] <- 1 - (ss_res / ss_tot)
    }
    
    # Store the average R^2 value for this feature
    results_df <- rbind(results_df, data.frame(feature = feat, avg_r_squared = mean(r_squared_values)))
  }
  
  # Arrange the results by avg_r_squared in descending order
  results_df <- results_df %>% dplyr::arrange(desc(avg_r_squared))
  
  return(results_df)
}


perform_cross_validation <- function(data, dependent_var, features) {
  results_df <- data.frame(feature = character(), avg_r_squared = numeric(), stringsAsFactors = FALSE)
  
  # Create a 10-fold cross-validation plan
  cv_folds <- createFolds(data[[dependent_var]], k = 10, list = TRUE, returnTrain = TRUE)
  
  for (feat in features) {
    r_squared_values <- numeric(length(cv_folds))
    
    fold_index <- 1
    for (train_indices in cv_folds) {
      # Split the data into training and testing based on the fold indices
      trainData <- data[train_indices, ]
      testData <- data[-train_indices, ]
      
      # Construct the formula
      formula_str <- as.formula(paste(dependent_var, "~", feat))
      
      # Fit the model on training data
      model <- lm(formula = formula_str, data = trainData)
      
      # Predict on testing data
      predictions <- predict(model, newdata = testData)
      
      # Calculate R^2
      ss_tot <- sum((testData[[dependent_var]] - mean(testData[[dependent_var]]))^2)
      ss_res <- sum((testData[[dependent_var]] - predictions)^2)
      r_squared_values[fold_index] <- 1 - (ss_res / ss_tot)
      
      fold_index <- fold_index + 1
    }
    
    # Store the average R^2 value for this feature
    results_df <- rbind(results_df, data.frame(feature = feat, avg_r_squared = mean(r_squared_values)))
  }
  
  # Arrange the results by avg_r_squared in descending order
  results_df <- results_df %>% arrange(desc(avg_r_squared))
  
  return(results_df)
}


## calculate r squared 
calculate_r_squared <- function(predicted, observed) {
  correlation_coefficient <- cor(predicted, observed)
  r_squared <- correlation_coefficient
  return(r_squared)
}

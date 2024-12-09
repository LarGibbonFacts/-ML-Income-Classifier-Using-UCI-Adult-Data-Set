#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table, recipes, dplyr, caret, glmnet)

# Set seed
set.seed(418518)

#####################
# Problem 1
#####################

# Set working directory
setwd("C:/Users/sweet/OneDrive/Documents/School/Fall 4/Econometrics")

# Load the data
data <- read.csv("ECON_418-518_HW3_Data.csv")

# Preprocess the data (adjusted factor levels for income)
data <- data %>%
  select(-fnlwgt, -occupation, -relationship, -capital.gain, -capital.loss, -educational.num) %>%
  mutate(
    income = factor(ifelse(income == ">50K", "High", "Low"), levels = c("Low", "High")),
    race = ifelse(race == "White", 1, 0),
    gender = ifelse(gender == "Male", 1, 0),
    workclass = ifelse(workclass == "Private", 1, 0),
    native.country = ifelse(native.country == "United-States", 1, 0),
    marital.status = ifelse(marital.status == "Married-civ-spouse", 1, 0),
    education = ifelse(education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0),
    age_sq = age^2,
    age = scale(age, center = TRUE, scale = TRUE),
    age_sq = scale(age_sq, center = TRUE, scale = TRUE),
    hours.per.week = scale(hours.per.week, center = TRUE, scale = TRUE)
  )

# Split the data into training and testing sets
set.seed(418518)
data <- data[sample(nrow(data)), ]
last_train_index <- floor(nrow(data) * 0.70)
training_data <- data[1:last_train_index, ]
testing_data <- data[(last_train_index + 1):nrow(data), ]

# Separate outcome variable and predictors
x_train <- as.matrix(training_data %>% select(-income))
y_train <- training_data$income
x_test <- as.matrix(testing_data %>% select(-income))
y_test <- testing_data$income

# Define lambda grid
lambda_grid <- 10^seq(5, -2, length = 50)

#####################
# Lasso Model (Before Filtering)
#####################
set.seed(418518)
lasso_model <- train(
  x = x_train, y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = trainControl(
    method = "cv",
    number = 10,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Predict on testing data and calculate classification accuracy
lasso_predictions <- predict(lasso_model, newdata = x_test)
lasso_accuracy_before <- mean(lasso_predictions == y_test)

#####################
# Ridge Model (Before Filtering)
#####################
set.seed(418518)
ridge_model <- train(
  x = x_train, y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),
  trControl = trainControl(
    method = "cv",
    number = 10,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Predict on testing data and calculate classification accuracy
ridge_predictions <- predict(ridge_model, newdata = x_test)
ridge_accuracy_before <- mean(ridge_predictions == y_test)

#####################
# Filtering Non-Zero Coefficients
#####################
lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
zero_coeff_vars <- rownames(as.matrix(lasso_coefficients))[as.matrix(lasso_coefficients) == 0]
non_zero_vars <- setdiff(colnames(x_train), zero_coeff_vars)

x_train_non_zero <- x_train[, non_zero_vars, drop = FALSE]
x_test_non_zero <- x_test[, non_zero_vars, drop = FALSE]

#####################
# Lasso Model (After Filtering)
#####################
set.seed(418518)
lasso_model_refit <- train(
  x = x_train_non_zero, y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = trainControl(
    method = "cv",
    number = 10,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Predict on testing data and calculate classification accuracy
lasso_predictions_refit <- predict(lasso_model_refit, newdata = x_test_non_zero)
lasso_accuracy_after <- mean(lasso_predictions_refit == y_test)

#####################
# Ridge Model (After Filtering)
#####################
set.seed(418518)
ridge_model_refit <- train(
  x = x_train_non_zero, y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),
  trControl = trainControl(
    method = "cv",
    number = 10,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Predict on testing data and calculate classification accuracy
ridge_predictions_refit <- predict(ridge_model_refit, newdata = x_test_non_zero)
ridge_accuracy_after <- mean(ridge_predictions_refit == y_test)

#####################
# Output Results
#####################
cat("\n### Classification Accuracy Rates ###\n")
cat("Lasso (Before Filtering):", round(lasso_accuracy_before * 100, 2), "%\n")
cat("Ridge (Before Filtering):", round(ridge_accuracy_before * 100, 2), "%\n")
cat("Lasso (After Filtering):", round(lasso_accuracy_after * 100, 2), "%\n")
cat("Ridge (After Filtering):", round(ridge_accuracy_after * 100, 2), "%\n")

install.packages("randomForest")
library(randomForest)

#####################
# Random Forest Model
#####################

# Load necessary packages
library(caret)
library(randomForest)

# Set up training control
train_control <- trainControl(
  method = "cv",  # 5-fold cross-validation
  number = 5
)

# Define grid for tuning
tune_grid <- expand.grid(
  mtry = c(2, 5, 9)  # Number of features tried at each split
)

# Define models with different numbers of trees
trees_list <- c(100, 200, 300)

# Evaluate models
rf_results <- list()

for (trees in trees_list) {
  set.seed(418518)
  rf_model <- train(
    income ~ .,  # Formula using "income" as outcome variable
    data = training_data,
    method = "rf",
    trControl = train_control,
    tuneGrid = tune_grid,
    ntree = trees  # Number of trees in the forest
  )
  rf_results[[paste0("rf_", trees)]] <- rf_model
}

# Combine all results and find the best model
all_rf_results <- do.call(rbind, lapply(rf_results, function(model) model$results))
all_rf_results$model <- rep(names(rf_results), each = nrow(tune_grid))  # Add model identifier

# Find the best model
best_rf_model <- all_rf_results[which.max(all_rf_results$Accuracy), ]

# Print the best model
cat("\n### Best Random Forest Model ###\n")
print(best_rf_model)


# Extract and summarize results for each model
for (trees in trees_list) {
  cat("\n### Random Forest with", trees, "trees ###\n")
  print(rf_results[[paste0("rf_", trees)]]$results)
}


cat("\nBest Random Forest Accuracy:", round(best_rf_model$Accuracy * 100, 2), "%\n")

# Compare with the best model from Part (v)
cat("Best Model from Part (v) Accuracy:\n")
cat("Lasso:", round(lasso_accuracy_before * 100, 2), "%\n")
cat("Ridge:", round(ridge_accuracy_before * 100, 2), "%\n")


# Predict on training data using the best random forest model
best_rf_model_name <- paste0("rf_", best_rf_model$mtry)
best_rf <- rf_results[[best_rf_model_name]]

rf_predictions <- predict(best_rf, newdata = training_data)

# Debug: Check the name of the best model
print(best_rf_model$model)  # Should return a valid name like "rf_100", "rf_200", etc.

# Extract the best random forest model
# Debug: Print all_rf_results to inspect available data
cat("\n### All RF Results ###\n")
print(all_rf_results)

# Retrieve the best model's configuration
best_rf_model_trees <- best_rf_model$model
best_rf_model_mtry <- best_rf_model$mtry

# Extract the best model based on trees and mtry
cat("\nBest RF Model: Trees =", best_rf_model_trees, ", mtry =", best_rf_model_mtry, "\n")
best_rf <- rf_results[[best_rf_model_trees]]

# Ensure the model exists
if (is.null(best_rf)) {
  stop("The best random forest model could not be retrieved. Check the model identifiers.")
}

# Predict on the training data
rf_predictions <- predict(best_rf, newdata = training_data)

# Generate confusion matrix
conf_matrix <- confusionMatrix(rf_predictions, training_data$income)

# Print confusion matrix and false positives/negatives
cat("\n### Confusion Matrix ###\n")
print(conf_matrix)

false_positives <- conf_matrix$table[2, 1]  # Predicted High, Actual Low
false_negatives <- conf_matrix$table[1, 2]  # Predicted Low, Actual High

cat("\nFalse Positives:", false_positives, "\n")
cat("False Negatives:", false_negatives, "\n")




#####################
# Evaluate Models on Testing Data
#####################

# Helper function to calculate classification metrics
calculate_metrics <- function(predictions, actuals) {
  # Create confusion matrix
  cm <- confusionMatrix(predictions, actuals)
  
  # Extract metrics
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]  # Precision
  recall <- cm$byClass["Sensitivity"]        # Recall
  f1_score <- 2 * (precision * recall) / (precision + recall)  # F1 Score
  
  # Return as a named vector
  return(c(
    accuracy = as.numeric(accuracy),
    precision = as.numeric(precision),
    recall = as.numeric(recall),
    f1_score = as.numeric(f1_score)
  ))
}

# Initialize results storage
model_results <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1_Score = numeric(),
  stringsAsFactors = FALSE
)

#####################
# Lasso Model
#####################
lasso_test_predictions <- predict(lasso_model, newdata = x_test)
lasso_metrics <- calculate_metrics(lasso_test_predictions, y_test)
model_results <- rbind(model_results, cbind(Model = "Lasso", t(lasso_metrics)))

#####################
# Ridge Model
#####################
ridge_test_predictions <- predict(ridge_model, newdata = x_test)
ridge_metrics <- calculate_metrics(ridge_test_predictions, y_test)
model_results <- rbind(model_results, cbind(Model = "Ridge", t(ridge_metrics)))

#####################
# Random Forest Models
#####################

# Loop through each random forest model
for (trees in trees_list) {
  rf_model <- rf_results[[paste0("rf_", trees)]]
  
  # Predict on testing data
  rf_test_predictions <- predict(rf_model, newdata = testing_data)
  
  # Calculate metrics
  rf_metrics <- calculate_metrics(rf_test_predictions, testing_data$income)
  model_results <- rbind(model_results, cbind(
    Model = paste0("Random Forest (", trees, " Trees)"), 
    t(rf_metrics)
  ))
}

#####################
# Combine Results
#####################
rownames(model_results) <- NULL  # Reset rownames for clean output

# Print Results
cat("\n### Classification Metrics for Each Model ###\n")
print(model_results)

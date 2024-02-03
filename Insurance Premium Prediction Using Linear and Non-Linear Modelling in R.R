#Loading the required libraries
library(randomForest)
library(ISLR2)
library(readr)
library(tidyverse)
library(class)
library(caret)
library(Metrics)
library(MASS)
library(gbm)
library(caret)
library(dplyr)

source('med_insurance.txt')

#################### K Nearest Neighbours #########################################################################################################################################

set.seed(1)

predictor_vars <- c("age", "sex","bmi", "children", "smoker", "region.northeast","region.northwest","region.southeast","region.southwest")
target_var <- "charges"

X <- clean_data_t[, predictor_vars] 
y <- clean_data_t[, target_var]

#Split data into train and tests
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
train_X <- X[train_index, ] 
train_y <- y[train_index] 
test_X <- X[-train_index, ] 
test_y <- y[-train_index] 

# Fit the KNN model and calculate RMSE for different values of k
k_values <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)

rmse_results <- data.frame(K = k_values, Train_RMSE = NA, Test_RMSE = NA)


for (k in k_values) {
  knnmodel <- knnreg(train_X,train_y,k=k)
  test_pred_y= predict(knnmodel, data.frame(test_X))
  train_pred_y = predict(knnmodel, data.frame(train_X))
  test_rmse = caret::RMSE(test_y, test_pred_y)
  train_rmse = caret::RMSE(train_y, train_pred_y)
  rmse_results[rmse_results$K == k, "Train_RMSE"] <- train_rmse
  rmse_results[rmse_results$K == k, "Test_RMSE"] <- test_rmse
}

print(rmse_results)

# Test RMSE vs Train RMSE for different values of k

my_plot <- ggplot(rmse_results, aes(x = K)) +
  geom_line(aes(y = Test_RMSE, color = "Test RMSE"), size = 1.5, show.legend = TRUE) +
  geom_point(aes(y = Test_RMSE, color = "Test RMSE"), size = 3, show.legend = FALSE) +
  geom_line(aes(y = Train_RMSE, color = "Train RMSE"), size = 1.5, show.legend = TRUE) +
  geom_point(aes(y = Train_RMSE, color = "Train RMSE"), size = 3, show.legend = FALSE) +
  xlab("K") +
  ylab("RMSE") +
  ggtitle("Test RMSE vs. Train RMSE for Different K values") +
  scale_color_manual(name = "RMSE Type", values = c("Test RMSE" = "darkred", "Train RMSE" = "darkblue"),
                     labels = c("Test RMSE", "Train RMSE")) +
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        legend.title = element_text(size = 12),
        legend.position = "top",
        legend.box.spacing = unit(0.2, "cm"),
        legend.text = element_text(size = 10),
        panel.grid.minor = element_blank())

print(my_plot)

#################### Bagging #########################################################################################################################################
source('med_insurance.txt')

#@Saiyam - Pls place your code here. 

































#################### RANDOM FOREST #########################################################################################################################################
source('med_insurance.txt')

set.seed(1)
#Split data into train and tests
train_indices <- sample(1:nrow(clean_data_t), 0.8 * nrow(clean_data_t))
train <- clean_data_t[train_indices, ]

#data for out-of-sample validation
test <- clean_data_t[-train_indices, ]


#iterating on randomforest
p=ncol(train)-1
mtryv = c(p,6,4)
ntreev = c(200,500,1000)
grid_val = expand.grid(mtryv,ntreev)
colnames(grid_val)=c('mtry','ntree')
n_trial = nrow(grid_val)
test_rmse = rep(0,n_trial)
train_rmse = rep(0,n_trial)
forest_fit = vector('list',n_trial)

for(i in 1:n_trial) 
{
  cat('RF Trial ',i,' out of ',n_trial,'\n')
  data.forest <- randomForest(charges ~ ., data = train, mtry = grid_val[i, 1], ntree = grid_val[i, 2])
  ifit = predict(data.forest)
  ofit=predict(data.forest,newdata=test)
  test_rmse[i] = sum((test$charges-ofit)^2)
  train_rmse[i] = sum((train$charges-ifit)^2)
  forest_fit[[i]]=data.forest
}
train_rmse = round(sqrt(train_rmse/nrow(train)),3); test_rmse = round(sqrt(test_rmse/nrow(test)),3)

#display losses for each combination of m and ntree
print(cbind(grid_val,test_rmse,train_rmse))

best_idx <- which.min(test_rmse)
best_mtry <- grid_val[best_idx,"mtry"]
best_ntree <- grid_val[best_idx,"ntree"]

best_test_rmse <- test_rmse[best_idx]
cat("The best mtry is ",best_mtry, "and the best ntree is ",best_ntree, "with a test mse of ",best_test_rmse)

#plot oob error using last fitted rf model which has the largest ntree.
par(mfrow=c(1,1))
plot(data.forest)

importance(data.forest)
varImpPlot(data.forest)


savehistory()
rm(list=ls())
#################### RANDOM FOREST with 10-fold Cross Validation #########################################################################################################################################

source('med_insurance.txt')

#Split data into train and tests
set.seed(1)
train_indices <- sample(1:nrow(clean_data_t), 0.8 * nrow(clean_data_t))
train <- clean_data_t[train_indices, ]

#data for out-of-sample validation
test <- clean_data_t[-train_indices, ]

#iterating on randomforest
p=ncol(train)-1
mtryv = c(p,6,4)
ntreev = c(200,500,1000)
grid_val = expand.grid(mtryv,ntreev)
colnames(grid_val)=c('mtry','ntree')
n_trial = nrow(grid_val)
forest_fit = vector('list',n_trial)

# Number of folds for k-fold cross-validation
k <- 10

# Initialize variables to store results
test_rmse <- matrix(0, nrow = n_trial, ncol = k)
training_rmse <- matrix(0, nrow = n_trial, ncol = k)

for (i in 1:n_trial) {
  cat('RF Trial ', i, ' out of ', n_trial, '\n')
  
  # Create random fold assignments for k-fold cross-validation
  parts <- sample(rep(1:k, length.out = nrow(clean_data_t)))
  
  for (cviter in 1:k) {
    # Split the data into training and testing sets for the current fold
    train_part <- clean_data_t[parts != cviter, ]
    test_part <- clean_data_t[parts == cviter, ]
    
    # Train the random forest model on the training fold
    data.forest <- randomForest(charges ~ ., data = train_part, mtry = grid_val[i, 1], ntree = grid_val[i, 2])
    
    # Make predictions on the testing set for the current fold
    ofit <- predict(data.forest, newdata = test_part)
    
    # Calculate the mean squared error for the current fold
    test_rmse[i, cviter] <- sqrt(mean((test_part$charges - ofit)^2))
    
    # Make predictions on the training set for the current fold
    ifit <- predict(data.forest, newdata = train_part)
    
    # Calculate the mean squared error for the training fold
    training_rmse[i, cviter] <- sqrt(mean((train_part$charges - ifit)^2))
  }
}

# Calculate the mean RMSE across all folds for each trial
mean_test_rmse <- round(apply(test_rmse, 1, mean),3)
mean_train_rmse <- round(apply(training_rmse, 1, mean),3)


# Find the index of the best combination of hyperparameters
best_idx <- which.min(mean_test_rmse)

# Display losses for each combination of mtry and ntree
results <- cbind(grid_val, mean_test_rmse, mean_train_rmse)
print(results)

# Plot test RMSE against the number of trees for the best mtry
best_mtry <- grid_val[best_idx, "mtry"]
best_ntree <- grid_val[best_idx,"ntree"]
best_test_rmse <- mean_test_rmse[best_idx]
cat("The best mtry is ",best_mtry, "and the best ntree is ",best_ntree, "with a test mse of ",best_test_rmse)

rm(list=ls())

#################### BOOSTING #########################################################################################################################################
source('med_insurance.txt')

# training vs testing data
set.seed(1)
train_indices <- sample(1:nrow(clean_data_t), 0.8 * nrow(clean_data_t))
train <- clean_data_t[train_indices, ]
test <- clean_data_t[-train_indices, ]

# tuning grid
grid <- expand.grid(n.trees = c(1500, 2000, 2500),
                    interaction.depth = c(20, 30),
                    shrinkage = c(0.05, 0.1, 0.15),
                    n.minobsinnode = c(5, 10))

# hyperparameter tuning with cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 5, summaryFunction = defaultSummary)

# parallel processing
library(doParallel)
registerDoParallel(cores = 4)

# train boosting model w/tuning grid
gbm_model <- train(charges ~ ., data = train, method = "gbm",
                   trControl = ctrl,
                   tuneGrid = grid,
                   distribution = "gaussian")

# best hyperparameters
print(gbm_model$bestTune)

# predictions on test data w/best model
predictions <- predict(gbm_model, newdata = test)

# RMSE
rmse <- sqrt(mean((test$charges - predictions)^2))
print(paste("RMSE:", rmse))

# R^2
actual_values <- test$charges
mean_actual <- mean(actual_values)
ss_total <- sum((actual_values - mean_actual)^2)
ss_residual <- sum((actual_values - predictions)^2)
r_squared <- 1 - (ss_residual / ss_total)
print(paste("R-squared:", r_squared))

# residuals
residuals <- test$charges - predictions

# residuals vs. predicted values
plot(predictions, residuals, main = "Residuals vs. Predicted Values", xlab = "Predicted Values", ylab = "Residuals")

rm(list=ls())

#################### Decision Tree #################################################
source('med_insurance.txt')

library(rpart)
library(tree)
library(rpart.plot)

# Define the number of folds for k-fold cross-validation
k = 10

# Initialize an empty vector to store RMSE values for each fold
rmse_values = numeric(k)

# Perform k-fold cross-validation
set.seed(1)  # For reproducibility
folds = cut(seq(1, nrow(clean_data_t)), breaks = k, labels = FALSE)
for (i in 1:k) {
  # Create training and testing datasets for the current fold
  test_indices = which(folds == i, arr.ind = TRUE)
  train_data = clean_data_t[-test_indices, ]
  test_data = clean_data_t[test_indices, ]
  
  # Fit the big tree using rpart.control on the training data
  # minsplit sets the minimum number of observations required in a node for it to be further split.
  big.tree = rpart(charges~., method = "anova", data = train_data,
                   control = rpart.control(minsplit = 10, cp = 0.0005))
  
  # Predict on the testing data
  predictions = predict(big.tree, newdata = test_data)
  
  # Calculate RMSE for the current fold
  rmse_values[i] = sqrt(mean((test_data$charges - predictions)^2))
}

# Calculate the mean RMSE across all folds
mean_rmse = mean(rmse_values)
cat('Mean RMSE for k-fold cross-validation: ', mean_rmse, '\n')

# Look at cross-validation for selecting best complexity parameter (cp)
big.tree <- rpart(charges~., method = "anova", data = clean_data_t,
                  control = rpart.control(minsplit = 10, cp = 0.0005))
nbig = length(unique(big.tree$where))
cat('size of big tree: ',nbig,'\n')
plotcp(big.tree)

# Select the best complexity parameter (cp) based on cross-validation results
bestcp <- big.tree$cptable[which.min(big.tree$cptable[, "xerror"]), "CP"]
cat('Best cp: ', bestcp, '\n')

# Plot the best-pruned tree based on the selected complexity parameter
best.tree <- prune(big.tree, cp = bestcp)
rpart.plot(best.tree)

rm(list=ls())

###################################################################
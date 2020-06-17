# SETUP -------------------------------------------------------------------

pacman::p_load(readr, titanic, dplyr, caret, randomForest)
train_all = read_csv("train.csv")

# TEST RANDOM SEEDS FOR SAMPLING ------------------------------------------
# Test random seeds for dividing data into training and validation sets

sampling_rows = 200000

sampling_results = tibble(seed = rep(NA, sampling_rows), 
                          train_survive = rep(NA, sampling_rows), 
                          train_die = rep(NA, sampling_rows), 
                          val_survive = rep(NA, sampling_rows), 
                          val_die = rep(NA, sampling_rows))


for (i in 1:sampling_rows) {
  
  if (i %% 5000 == 0) {print(i)}
  
  set.seed(i)
  train_indices = sample(1:nrow(train_all), round(0.8 * nrow(train_all)))
  
  train = train_all[train_indices,]
  val = train_all[-train_indices,]
  
  sampling_results$seed[i] = i
  sampling_results$train_survive[i] = sum(train$Survived == 1) / nrow(train)
  sampling_results$train_die[i] = sum(train$Survived == 0) / nrow(train)
  
  sampling_results$val_survive[i] = sum(val$Survived == 1) / nrow(val)
  sampling_results$val_die[i] = sum(val$Survived == 0) / nrow(val)
  
}

# VIEW AND SAVE RESULTS ---------------------------------------------------

quantile(sampling_results$train_survive)
quantile(sampling_results$train_die)
quantile(sampling_results$val_survive)
quantile(sampling_results$val_die)

write_csv(sampling_results, paste0("../r_sampling_results_", sampling_rows, ".csv"))

# STRATIFIED SAMPLING -----------------------------------------------------

# Change fields to factor to allow random forest model to work
train_all = mutate(train_all, 
                   Survived = as.factor(Survived), 
                   Sex = as.factor(Sex), 
                   Embarked = as.factor(Embarked))

set.seed(20200226)
strat_indices = createDataPartition(train_all$Survived, p = 0.8)
train = train_all[strat_indices$Resample1,]
val = train_all[-strat_indices$Resample1,]

# MODELING ----------------------------------------------------------------
# Test random seeds for modeling

model = Survived ~ Pclass + Sex + SibSp + Fare  # Specify model

model_rows = 25000

# Create dataframe for model results
model_results = tibble(seed = rep(NA, model_rows), 
                       acc = rep(NA, model_rows), 
                       prec = rep(NA, model_rows), 
                       recall = rep(NA, model_rows))

for (i in 1:model_rows) {
  
if (i %% 1000 == 0) {print(i)}

set.seed(i)  # Set seed

# Create random forest
rf = randomForest(model, data = train, ntree = 50)

pred_holder = mutate(val, 
              pred = predict(rf, newdata = select(val, Pclass, Sex, SibSp, Fare)))  # Generate validation set predictions

# Fill in "model_results"
model_results$seed[i] = i
model_results$acc[i] = sum(pred_holder$Survived == pred_holder$pred) / nrow(pred_holder)

prec_holder = filter(pred_holder, pred == 1) 
model_results$prec[i] = sum(prec_holder$Survived == prec_holder$pred) / nrow(prec_holder)

recall_holder = filter(pred_holder, Survived == 1)
model_results$recall[i] = sum(recall_holder$Survived == recall_holder$pred) / nrow(recall_holder)
  
rm(pred_holder, prec_holder, recall_holder, rf)

}

quantile(model_results$acc)
quantile(model_results$prec)
quantile(model_results$recall)

write_csv(model_results, paste0("../r_model_results_", model_rows, ".csv"))
library(ggplot2)
library(dplyr)
library(gridExtra)
library(MASS)
library(crossval)
library(class)
library(randomForest)
library(tidyr)

# ----------------------- EDA ------------------------------
dataset = read.csv("D:/Statistica Applicata/hearth disease failure/heart_failure_clinical_records.csv")
summary(dataset)
head(dataset)

dataset$anaemia = as.factor(dataset$anaemia)
dataset$diabetes = as.factor(dataset$diabetes)
dataset$high_blood_pressure = as.factor(dataset$high_blood_pressure)
dataset$sex = as.factor(dataset$sex)
dataset$smoking = as.factor(dataset$smoking)
dataset$DEATH_EVENT = as.factor(dataset$DEATH_EVENT)
summary(dataset)

ggplot(data = dataset, aes(x=DEATH_EVENT)) + geom_bar()


dataset %>% select_if(function(x) is.factor(x)) %>% gather() %>% ggplot(aes(x=value)) + facet_wrap(~ key, scales="free") + geom_bar()
dataset %>% select_if(function(x) is.numeric(x)) %>% gather() %>% ggplot(aes(x=value, y = after_stat(density))) + facet_wrap(~ key, scales="free") + geom_histogram() + geom_density(lwd=1)
# Given that we would like to apply Linear Discriminant Analysis model, we will try to normalize as much as possible the explanatory numerical variables

dataset$log_creatinine_phosphokinase = log(dataset$creatinine_phosphokinase)
dataset$log_serum_creatinine = log(dataset$serum_creatinine)

dataset = dataset %>% dplyr::select(-c(creatinine_phosphokinase,serum_creatinine))
summary(dataset)
dataset %>% select_if(function(x) is.numeric(x)) %>% gather() %>% ggplot(aes(x=value, y = after_stat(density))) + facet_wrap(~ key, scales="free") + geom_histogram() + geom_density(lwd=1)


# Plot the barplots of every categorical variable, wrt DEATH_EVENT
dataset %>% select_if(function(x) is.factor(x)) %>% gather(key, value, -DEATH_EVENT) %>% ggplot(aes(x=value, y=..count..)) + facet_wrap(~ key, scales="free") + geom_bar(aes(fill=DEATH_EVENT), position = "dodge")


# Plot the boxplots of every numerical variable, wrt DEATH_EVENT
dataset_numeric = dataset %>% select_if(function(x) is.numeric(x))
dataset_numeric$DEATH_EVENT = dataset$DEATH_EVENT
summary(dataset_numeric)

age_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=age)) + geom_boxplot()
ejection_fraction_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=ejection_fraction)) + geom_boxplot()
platelets_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=platelets)) + geom_boxplot()
serum_sodium_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=serum_sodium)) + geom_boxplot()
time_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=time)) + geom_boxplot()
log_creatinine_phosphokinase_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=log_creatinine_phosphokinase)) + geom_boxplot()
log_serum_creatinine_death = ggplot(data=dataset_numeric, aes(x=DEATH_EVENT, y=log_serum_creatinine)) + geom_boxplot()

grid.arrange(age_death, ejection_fraction_death, platelets_death, serum_sodium_death, time_death, log_creatinine_phosphokinase_death, log_serum_creatinine_death)


# Given the above plots, I remove log_creatinine_phosphokinase and platelets

simple_dataset = dataset %>% dplyr::select(-c(log_creatinine_phosphokinase,platelets))
summary(simple_dataset)


# ----------------------- Create a train-test split -------------------------------------
train_test_split = function(dataset, prop = 0.7){
  prop_idx = nrow(dataset) * prop
  train_split = dataset %>% slice(1:prop_idx)
  test_split = dataset %>% slice((prop_idx+1):nrow(dataset))
  return(list("train" = train_split, "test" = test_split))
}

split = train_test_split(simple_dataset)
trainset = data.frame(split["train"])
colnames(trainset) = colnames(simple_dataset)
testset = data.frame(split["test"])
colnames(testset) = colnames(simple_dataset)

# --------------------------------------     FEATURE SELECTION for GLM ----------------------------------------------

model.lr = glm(DEATH_EVENT ~ ., data = trainset, family=binomial(link="logit"))
summary(model.lr)
model.lr$deviance / model.lr$null.deviance

model_simpler.lr = glm(DEATH_EVENT ~ . - anaemia, data = trainset, family=binomial(link="logit"))
summary(model_simpler.lr)
model_simpler.lr$deviance / model_simpler.lr$null.deviance

model_simpler.lr = glm(DEATH_EVENT ~ . - smoking - anaemia - diabetes, data = trainset, family=binomial(link="logit"))
summary(model_simpler.lr)
model_simpler.lr$deviance / model_simpler.lr$null.deviance

model_simpler.lr = glm(DEATH_EVENT ~ . - smoking - anaemia - diabetes - sex, data = trainset, family=binomial(link="logit"))
summary(model_simpler.lr)
model_simpler.lr$deviance / model_simpler.lr$null.deviance

# CHECK GLM ASSUMPTIONS
# Assumptions are blatently not respected, for this reason we DO NOT CONSIDEER GLM
par(mfrow = c(2,2))
plot(model_simpler.lr, which=1)
plot(model_simpler.lr, which=2)
plot(model_simpler.lr, which=3)
plot(model_simpler.lr, which=4)
par(mfrow=c(1,1))

# -------------------------------------------- LDA, KNN and Random Forest ---------------------------------------------

# LDA Model
model.lda = lda(DEATH_EVENT ~ . - smoking - anaemia - diabetes - time, data = trainset)
model.lda

prediction.lda = predict(model.lda, testset)$class

# K-NEAREST-NEIGHBOUR
model.knn = knn(train = trainset, test = testset, cl = trainset$DEATH_EVENT, k=2)
summary(model.knn)

cm.knn = table(model.knn, testset$DEATH_EVENT)
# RANDOM FOREST

model.rf = randomForest(data = trainset, DEATH_EVENT ~ . - smoking - anaemia - diabetes, mtry = sqrt(ncol(trainset)), ntree = 500)
model.rf

prediction.rf = predict(model.rf, testset)

#  ---------------------------------- MODEL SELECTION -------------------------------------------

# Given that the response variable is class unbalanced, then I use F1-Score, instead of Accuracy
f1_score <- function(cm){
  precision = (cm["1","1"]) / (cm["1","1"] + cm["0","1"])
  recall = (cm["1","1"]) / (cm["1","1"] + cm["1","0"])
  return(2*(precision*recall)/(precision+recall))
}

cm.lda = table(prediction.lda, testset$DEATH_EVENT)
f1_score(cm.lda)

f1_score(cm.knn)

cm.rf = table(prediction.rf, testset$DEATH_EVENT)
f1_score(cm.rf)

# ------------------------------ CONCLUSION -----------------------------------------
# The best model is Random Forest, with 99.89% f1-score

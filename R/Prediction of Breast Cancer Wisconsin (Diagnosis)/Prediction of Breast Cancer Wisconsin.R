### Import the required libraries

library(mlbench)
library(corrplot)
library(psych)
library(e1071)
library(caret)
library(DataExplorer)
library(ggplot2)
library(tibble)
library(factoextra)
library(kernlab)
library(MASS)
library(mda)
library(earth)
library(klaR)
library(glmnet)

### Import the data set

cancer = read.csv("data.csv",sep=",",header = TRUE)
head(cancer)
dim(cancer)
names(cancer)

### Data Pre-processing

# Interpretation missing values

str(cancer)                             # to see the column names and data type
plot_missing(cancer)                    # plot to visualize the missing data
cancer = cancer[2:32]                   # selecting the columns 2 to 32
dim(cancer)

summary(cancer)
colSums(is.na(cancer))                  # to see the null values in the data set

# Frequency distribution of categorical variable (diagnosis)

ggplot(cancer, aes(x = factor(diagnosis))) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) + 
  scale_y_continuous(labels=scales::percent) + 
  ggtitle("Relative frequency distribution of the Diagnosis variable") +
  ylab("Relative Frequencies") + 
  theme_classic()

freq_region <- table(cancer$diagnosis)
freq_region
barplot(freq_region, main="Frequency distribution of the Diagnosis predictor")

# No dummy variables are necessary and all are real valued, so no need of near zero variance.

# Interpretation of Outliers and Skewness

par(mfrow = c(3, 5))

for (i in 2:ncol(cancer)) {
  boxplot(cancer[ ,i], xlab = names(cancer[i]), 
          main = paste("Boxplot of", names(cancer[i])))  
}

dev.off()

par(mfrow = c(3, 5))

for (i in 2:ncol(cancer)) {
  plot(density(cancer[ ,i]), main = paste("Density", names(cancer[i])))
}

dev.off()

skewValues <- apply(cancer[,2:31], 2, skewness)
skewValues                             #skewness of each predictor

# Applying transformations

cancer_pp <- preProcess(cancer, method = c("BoxCox", "center", "scale", "spatialSign"))
cancer_trans <- predict(cancer_pp, cancer)

# Checking outliers and skewness after transformation

par(mfrow = c(3, 5))

for (i in 2:ncol(cancer_trans)) {
  boxplot(cancer_trans[ ,i], xlab = names(cancer_trans[i]), main = paste("Boxplot of", names(cancer_trans[i])))  
}

dev.off()

par(mfrow = c(3, 5))

for (i in 2:ncol(cancer_trans)) {
  plot(density(cancer_trans[ ,i]), main = paste("Density", names(cancer_trans[i])))
}

dev.off()

# correlation
correlations <- cor(cancer[,2:31])
dim(cancer)

corrplot(correlations, order = "hclust") # to visualize the clusters of highly correlation predictors
corPlot(cancer[,2:31], cex = 0.5)               # to visualize the values of highly correlation predictors

highCorr_75 <- findCorrelation(correlations, .75)
length(highCorr_75)
data_lt75 <- cancer[,-highCorr_75]
dim(data_lt75)

highCorr_85 <- findCorrelation(correlations, .85)
length(highCorr_85)
data_lt85 <- cancer[,-highCorr_85]
dim(data_lt85)

highCorr_90 <- findCorrelation(correlations, .90)
length(highCorr_90)
data_lt90 <- cancer[,-highCorr_90]
dim(data_lt90)
colnames(data_lt90)     # predictors which have correlation less than 0.9

highCorr_90 <- colnames(cancer)[findCorrelation(correlations, .90)]
highCorr_90             # predictors which have correlation greater than 0.9


# Removing the 10 predictors which are highly correlated
cancer_new <- cancer[, which(!colnames(cancer) %in% highCorr_90)]
dim(cancer_new)
str(cancer_new)

# Checking proportion of variance explained after removing the highly correlated 10 predictors
cancer_pca <- prcomp(cancer_new, center=TRUE, scale=TRUE)
summary(cancer_pca)

pca_var <- cancer_pca$sdev^2
pve_df <- pca_var / sum(pca_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(cancer_new)), pve_df, cum_pve)

ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "blue", slope = 0)

fviz_screeplot(cancer_pca)

### Data Splitting

set.seed(1)

cancer_df <- cbind(diagnosis = cancer$diagnosis, cancer_new)
trainingRows <- createDataPartition(cancer_df$diagnosis, p = 0.8, list = FALSE)
nrow(trainingRows)

X_train <- cancer_df[trainingRows, ]
X_test <- cancer_df[-trainingRows, ]
y_train <- data.frame(as.factor(cancer_df$diagnosis))[trainingRows, ]
y_test <- data.frame(as.factor(cancer_df$diagnosis))[-trainingRows, ]

## linear classification models

# logistic regression

ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE, savePredictions = TRUE)
set.seed(1)
glm <- train(diagnosis~.,
             data = X_train,
             method = "glm",
             family="binomial",
             trControl = ctrl)

glm
glmPred <- predict(glm, newdata = X_test)
confusionMatrix(data=glmPred, reference=y_test)

# # multinomial model
# 
# ctrl <- trainControl(method = "cv", number = 5,
#                      classProbs = TRUE, savePredictions = TRUE)
# mnGrid <- expand.grid(.decay = c(0.1:10))
# set.seed(1)
# mn <- train(diagnosis~.,
#              data = X_train,
#              method = "multinom",
#              preProc = c("center", "scale"),
#              metric = "Kappa",
#              tuneGrid = mnGrid,
#              trControl = ctrl)
# mn
# plot(mn)
# 
# mnPred <- predict(mn, newdata = X_test)
# mnPred_df <- data.frame(obs = y_test, pred = mnPred)
# colnames(mnPred_df) <- c("obs","pred")
# confusionMatrix(data=mnPred, reference=y_test)

# Linear Discriminant Analysis
set.seed(100)
lda <- train(diagnosis~.,
             data = X_train, method = "lda", trControl = ctrl, 
             preProc = c("center","scale"), metric = "Kappa")
lda

# prediction
ldaPred <- predict(lda, newdata = X_test)
confusionMatrix(data=ldaPred, reference=y_test)

## Partial Least Squares Discriminant Analysis
set.seed(1)
plsda <- train(diagnosis~.,
               data = X_train, method = "pls", tuneLength = 30, 
               preProcess = c("center","scale"), metric = "Kappa", 
               trControl = ctrl)
plsda

plot(plsda, main="Tuning paramter plot of PLSDA model")

# predictions
plsPred <- predict(plsda, newdata = X_test, ncomp = 1)
confusionMatrix(data=plsPred, reference=y_test)

## penalized models

# glmnet
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

set.seed(100)
glmnTuned <- train(diagnosis~.,
                   data = X_train,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "Kappa",
                   trControl = ctrl)
glmnTuned
plot(glmnTuned, main="Tuning parameter plot of GLMNET model")

# predictions
glmnPred <- predict(glmnTuned, newdata = X_test)
confusionMatrix(data=glmnPred, reference=y_test)

# sparseLDA
library(sparseLDA)
sparseGrid <- expand.grid(.lambda = seq(0.01, 0.1, 0.2), .NumVars = c(-4:4))

set.seed(1)
sparseTuned <- train(diagnosis~.,
                     data = X_train,
                     method = "sparseLDA",
                     tuneGrid = sparseGrid,
                     preProc = c("center", "scale"),
                     importance = TRUE,
                     metric = "Kappa",
                     trControl = ctrl)

sparseTuned
plot(sparseTuned,  main="Tuning parameter plot of sparseLDA model")

# predictions
sparsePred <- predict(sparseTuned, newdata = X_test)
confusionMatrix(data=sparsePred, reference=y_test)


## Non-linear classification models

# multiple discriminant analysis
library(mda)

set.seed(1)
mda <- train(diagnosis~.,
             data = X_train,
             method = "mda",
             preProc = c("center", "scale"),
             metric = "Kappa",
             tuneGrid = expand.grid(.subclasses = 1:10),
             trControl = ctrl)
mda
plot(mda, main="Tuning parameter plot of MDA model")

# predictions
mdaPred <- predict(mda, newdata = X_test)
confusionMatrix(data=mdaPred, reference=y_test)

# rda

set.seed(1)
rdaGrid = expand.grid(.gamma = c(0, 0.01, .1, 0.2), .lambda = seq(0.01, 0.2, length=10))

rda <- train(diagnosis~.,
             data = X_train,
             method = "rda",
             preProc = c("center", "scale"),
             metric = "Kappa",
             tuneGrid = rdaGrid,
             trControl = ctrl)
rda
plot(rda, main="Tuning parameter plot of RDA model")

# predictions
rdaPred <- predict(rda, newdata = X_test)
confusionMatrix(data=rdaPred, reference=y_test)

# neural network
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
set.seed(1)
ctrl <- trainControl(method = "cv", number = 10, 
                     classProbs = TRUE, savePredictions = TRUE)

nnetTune <- train(diagnosis~.,
                  data = X_train,
                  method = "nnet",
                  tuneGrid = nnetGrid,
                  preProc = c("center", "scale"),
                  metric = "Kappa",
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(X_train) + 1) + 10 + 1,
                  maxit = 2000,
                  trControl = ctrl)
nnetTune
plot(nnetTune, main="Tuning parameter plot of Neural Network model")

nnetPred <- predict(nnetTune, newdata = X_test)
confusionMatrix(data=nnetPred, reference=y_test)

# SVM model
# library(dplyr)
# y_train %>% 
#   filter_all(any_vars(is.na(.)))      # to filter na values

set.seed(1)
svmRTuned <- train(diagnosis~.,
                   data = X_train,
                   method = "svmRadial",
                   tuneLength = 14,
                   preProc = c("center", "scale"),
                   metric = "Kappa",
                   trControl = ctrl)
svmRTuned
ggplot(svmRTuned)+coord_trans(x='log2') # use log2 scale

svmPred <- predict(svmRTuned, newdata = X_test)
confusionMatrix(data=svmPred, reference=y_test)

# fda

fdaGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
set.seed(1)
fda <- train(diagnosis~.,
             data = X_train,
             method = "fda",
             preProc = c("center", "scale"),
             metric = "Kappa",
             tuneGrid = fdaGrid,
             trControl = ctrl)
fda
plot(fda, main="Tuning parameter plot of FDA model")

fdaPred <- predict(fda, newdata = X_test)
confusionMatrix(data=fdaPred, reference=y_test)

# knn
library(caret)
set.seed(1)
knnFit <- train(diagnosis~.,
                data = X_train,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:50),
                trControl = ctrl)

knnFit
plot(knnFit, main="Tuning parameter plot of KNN model")
knnPred <- predict(knnFit, newdata = X_test)
confusionMatrix(data=knnPred, reference=y_test)

# naive bayes

set.seed(1)
nbFit <- train( diagnosis~.,
                data = X_train,
                method = "nb",
                metric = "kappa",
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)
nbFit

nbPred <- predict(nbFit, newdata = X_test)
confusionMatrix(data = nbPred, reference =y_test)

# random forest
set.seed(1)
rfTuned <- train(diagnosis~.,
                 data = X_train,
                 method = "rf",
                 preProc = c("center", "scale"),
                 tunelength = 15,
                 metric = "Kappa",
                 trControl = ctrl)
rfTuned
plot(rfTuned, main="Tuning parameter plot of Random Forest model")

rfPred <- predict(rfTuned, newdata = X_test)
confusionMatrix(data=rfPred, reference=y_test)


#####################################

# Top Predictors
varImp(nnetTune)
nnet_varimp <- varImp(nnetTune)
plot(nnet_varimp, 5)









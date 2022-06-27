## Objective
The study of the Breast Cancer Wisconsin (Diagnosis) has the characteristics of cell nuclei to understand its structure and its change. The objective of the project is to predict whether the tumor is benign (not cancer) or malign (cancer) which implies that it is a two-class classification problem.

![image](https://user-images.githubusercontent.com/108242990/175839594-9f7acaac-2edb-419a-9d8f-5cfecbfc8a18.png)

## Project Description
To predict whether the tumor is benign or malign, we have taken a dataset from the UCI repository. We performed data pre-processing steps for skewness, outliers, missing values, and duplicate variables. We applied a box-cox transformation to mitigate the problem of skewness, and spatial sign transformation for outliers and removed highly correlated variables with a cutoff of 0.90. Post-pre-processing, we are left with 21 predictors and 1 response variable. The data split in standard notion 80:20 with applying stratified random sampling technique. The models developed for both linear and non-classification models, applying 10-fold cross-validation as a resampling technique, selected the best two models from each section based on statistic measure kappa. The best four models are tested on test data and finalized the neural network model achieving the highest kappa value of 0.8852 as compared to the other three models.

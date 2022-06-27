
### Import the required libraries ###

library(car)
library(olsrr)
library(tables)
library(xtable)
library(psych)
library(PerformanceAnalytics)
library(corrplot)
library(plyr)


### Import the data set ###

senic_df = read.table("SENIC.csv",sep=",",header = TRUE)
head(senic_df)

# Boxplot of response variable

boxplot(senic_df$Y,xlab="Average Length of Stay(Y)", main="Boxplot of Response Variable")

### Exploratory Data Analysis ###

# Histogram of the variables

par(mfrow=c(3,4))
hist(senic_df$Y,xlab="Avg Length of Stay(Y)", main="Histogram of Y")
hist(senic_df$X1,xlab="Avg age of patients(X1)", main="Histogram of X1")
hist(senic_df$X2,xlab="Infection Risk(X2)", main="Histogram of X2")
hist(senic_df$X3,xlab="Routine Culturing Ratio(X3)", main="Histogram of X3")
hist(senic_df$X4,xlab="Routine X-ray Ratio(X4)", main="Histogram of X4")
hist(senic_df$X5,xlab="Number of beds in hospital(X5)", main="Histogram of X5")
hist(senic_df$X6,xlab="Medical School Association(X6)", main="Histogram of X6")
hist(senic_df$X7,xlab="Geographic Region of Hospital(X7)", main="Histogram of X7")
hist(senic_df$X8,xlab="Average Census(X8)", main="Histogram of X8")
hist(senic_df$X9,xlab="Registered and Licensed Practical Nurses(X9)", main="Histogram of X9")
hist(senic_df$X10,xlab="Available Facilities and Services(X10)", main="Histogram of X10")

# Boxplots of the variables

par(mfrow=c(3,4))
boxplot(senic_df$Y,xlab="Avg Length of Stay(Y)", main="Boxplot of Y")
boxplot(senic_df$X1,xlab="Avg age of patients(X1)", main="Boxplot of X1")
boxplot(senic_df$X2,xlab="Infection Risk(X2)", main="Boxplot of X2")
boxplot(senic_df$X3,xlab="Routine Culturing Ratio(X3)", main="Boxplot of X3")
boxplot(senic_df$X4,xlab="Routine X-ray Ratio(X4)", main="Boxplot of X4")
boxplot(senic_df$X5,xlab="Number of beds in hospital(X5)", main="Boxplot of X5")
boxplot(senic_df$X6,xlab="Medical School Association(X6)", main="Boxplot of X6")
boxplot(senic_df$X7,xlab="Geographic Region of Hospital(X7)", main="Boxplot of X7")
boxplot(senic_df$X8,xlab="Average Census(X8)", main="Boxplot of X8")
boxplot(senic_df$X9,xlab="Registered and Licensed Practical Nurses(X9)", main="Boxplot of X9")
boxplot(senic_df$X10,xlab="Available Facilities and Services(X10)", main="Boxplot of X10")
dev.off()

# Summary Statistics

summary(senic_df)

# Scatter Plot Matrix

pairs(senic_df)

# Added-Variable Plots

senic_df.lmfit <- lm(Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, data=senic_df)
avPlots(senic_df.lmfit)

# correlation matrix

cor(senic_df)
# chart.Correlation(senic_df, histogram = TRUE, method = "pearson")

corPlot(senic_df, cex = 1.2)
# corrplot.mixed(cor(senic_df), lower = "number", upper = "circle", tl.col = "black")


# To verify the NA values exists in the data set
tabular(mean + sd ~ Y + X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, data=senic_df)

# To check the frequency distribution of the geographic region for hospital(X7)
freq_region <- table(senic_df$X7)
freq_region
barplot(freq_region, main="frequency distribution of the geographic region for hospital")


### Model Fitting ###

senic_df.lmfit

senic_df.lmfit2 <- lm(Y ~ 1, data=senic_df)   # Intercept only model
summary(senic_df.lmfit2)

# Summary of the full model
summary(senic_df.lmfit)

# ANOVA Table of the full model
anova(senic_df.lmfit)

# Reduced Model

lmfit_red <- lm(Y ~ X1 + X2 + X7 + X8 + X9, data=senic_df)
summary(lmfit_red)

### Model Selection ###

# Load HH, leaps, and StepReg packages

library(leaps)
library(HH)
library(StepReg)
library(lmtest)

# Adjusted R2

b = bestsubset(data=senic_df, y="Y", select="adjRsq", best=5)
print(b)
stepwise(data=senic_df ,y="Y", select="adjRsq")
plot(b[,1:2])

# Cp

b = bestsubset(data=senic_df,y="Y",select="CP",best=5)
print(b)
stepwise(data=senic_df,y="Y",select="CP")
plot(b[,1:2])

# AIC

b = bestsubset(data=senic_df,y="Y",select="AIC",best=5)
print(b)
stepwise(data=senic_df,y="Y",select="AIC")
plot(b[,1:2])

# BIC

b = bestsubset(data=senic_df,y="Y",select="BIC",best=5)
print(b)
stepwise(data=senic_df,y="Y",select="BIC")
plot(b[,1:2])

# Final Model

reduced.lmfit <- lm(Y ~ X2 + X7 + X8 + X9 + X1, data=senic_df)
summary(reduced.lmfit)

reduced.lmfit2 <- lm(Y ~ X2 + X7 + X8 + X9 + X1 + X4 + X5, data=senic_df)
summary(reduced.lmfit2)

reduced.lmfit3 <- lm(Y ~ X2 + X7 + X8 + X9 + X1 + X5, data=senic_df)
summary(reduced.lmfit3)

### Regression Diagnostics ###

# Hat matrix and leverages

X <- as.matrix(cbind(1,senic_df[,-1]))
inv.XX <- solve(t(X)%*%X)
H <- X%*%inv.XX%*%t(X)
lev <- diag(H)

res <- rstudent(reduced.lmfit2)
fitted.y <- fitted(reduced.lmfit2)

# Residual Plots

par(mfrow=c(3,3))
plot(res ~ senic_df$X2, xlab="X2", ylab="Residual", main="Residuals vs. X2")
abline(h=0)
plot(res ~ senic_df$X7, xlab="X7", ylab="Residual", main="Residuals vs. X7")
abline(h=0)
plot(res ~ senic_df$X8, xlab="X8", ylab="Residual", main="Residuals vs. X8")
abline(h=0)
plot(res ~ senic_df$X9, xlab="X9", ylab="Residual", main="Residuals vs. X9")
abline(h=0)
plot(res ~ senic_df$X1, xlab="X1", ylab="Residual", main="Residuals vs. X1")
abline(h=0)
plot(res ~ senic_df$X4, xlab="X4", ylab="Residual", main="Residuals vs. X4")
abline(h=0)
plot(res ~ senic_df$X5, xlab="X5", ylab="Residual", main="Residuals vs. X5")
abline(h=0)

dev.off()
plot(res ~ fitted.y, xlab="Fitted value", ylab="Residual", main="Residuals vs. Fitted Values")
abline(h=0)

# Constancy of Error Variances 

bptest(reduced.lmfit2)

# Assumption of non-independence
plot(res, xlab="index | time (observation numbers)", ylab="residual"
     , main="Plot of jackknifed residual against index" )
abline(h=0,lwd=2)

#  Multi-collinearity 

vif(reduced.lmfit2)

# Checking the normality of the error terms

qqnorm(res);qqline(res)
shapiro.test(res)

## Checking whether there exist outliers in the data

p <- ncol(X)
n <- nrow(reduced.lmfit2)
which(lev > 2*p/n)
leveragePlots(reduced.lmfit2)
hist(lev, main="Histogram of leverage points", xlab="leverage")
abline(v=2*p/n, lty=2, lwd=2, col="red")

# Detection methods of influential points
# DFFITS
ols_plot_dffits(reduced.lmfit2)
dffits(reduced.lmfit2)

# Cook's D
ols_plot_cooksd_chart(reduced.lmfit2)
cooks.distance(reduced.lmfit2)

# DFBETAS
ols_plot_dfbetas(reduced.lmfit2)
dfbetas(reduced.lmfit2)

### Model Transformation ###
library(EnvStats)

boxcox.summary <- boxcox(reduced.lmfit2, optimize=TRUE)
lambda <- boxcox.summary$lambda
lambda
trans.Y <- senic_df$Y^lambda

senic_df <- cbind(senic_df,trans.Y)

## Re-fitting a model using the transformed response variable

boxcox.lmfit <- lm(trans.Y ~ X2 + X7 + X8 + X9 + X1 + X4 + X5, data=senic_df)
summary(boxcox.lmfit)

boxcox.res <- rstudent(boxcox.lmfit)

boxcox.fitted.y <- fitted(boxcox.lmfit)

# Residual Plots

par(mfrow=c(3,3))
plot(boxcox.res ~ senic_df$X2, xlab="X2", ylab="Residual", main="Residuals vs. X2")
abline(h=0)
plot(boxcox.res ~ senic_df$X7, xlab="X7", ylab="Residual", main="Residuals vs. X7")
abline(h=0)
plot(boxcox.res ~ senic_df$X8, xlab="X8", ylab="Residual", main="Residuals vs. X8")
abline(h=0)
plot(boxcox.res ~ senic_df$X9, xlab="X9", ylab="Residual", main="Residuals vs. X9")
abline(h=0)
plot(boxcox.res ~ senic_df$X1, xlab="X1", ylab="Residual", main="Residuals vs. X1")
abline(h=0)
plot(boxcox.res ~ senic_df$X4, xlab="X4", ylab="Residual", main="Residuals vs. X4")
abline(h=0)
plot(boxcox.res ~ senic_df$X5, xlab="X5", ylab="Residual", main="Residuals vs. X5")
abline(h=0)

dev.off()

plot(boxcox.res ~ boxcox.fitted.y, xlab="Fitted value", ylab="Residual", main="Residuals vs. Fitted Values")
abline(h=0)

# Constancy of Error Variances 

bptest(boxcox.lmfit)

# Multicollinearity 

vif(boxcox.lmfit)

# Normality 
qqnorm(boxcox.res);qqline(boxcox.res)
shapiro.test(boxcox.res)

### Final Model ###

final.lmfit <- boxcox.lmfit
summary(final.lmfit)
anova(final.lmfit)




### Check by removing X4 and X5 ###

reduced.lmfit <- lm(Y ~ X2 + X7 + X8 + X9 + X1, data=senic_df)
summary(reduced.lmfit)

### Regression Diagnostics ###

# Hat matrix and leverages

X <- as.matrix(cbind(1,senic_df[,-1]))
inv.XX <- solve(t(X)%*%X)
H <- X%*%inv.XX%*%t(X)
lev <- diag(H)

res <- rstudent(reduced.lmfit)
fitted.y <- fitted(reduced.lmfit)

# Residual Plots

par(mfrow=c(3,2))
plot(res ~ senic_df$X2, xlab="X2", ylab="Residual", main="Residuals vs. X2")
abline(h=0)
plot(res ~ senic_df$X7, xlab="X7", ylab="Residual", main="Residuals vs. X7")
abline(h=0)
plot(res ~ senic_df$X8, xlab="X8", ylab="Residual", main="Residuals vs. X8")
abline(h=0)
plot(res ~ senic_df$X9, xlab="X9", ylab="Residual", main="Residuals vs. X9")
abline(h=0)
plot(res ~ senic_df$X1, xlab="X1", ylab="Residual", main="Residuals vs. X1")
abline(h=0)


dev.off()
plot(res ~ fitted.y, xlab="Fitted value", ylab="Residual", main="Residuals vs. Fitted Values")
abline(h=0)

# Constancy of Error Variances 

bptest(reduced.lmfit)

#  Multi-collinearity 

vif(reduced.lmfit)

# Checking the normality of the error terms

qqnorm(res);qqline(res)
shapiro.test(res)

## Checking whether there exist outliers in the data

p <- ncol(X)
n <- nrow(reduced.lmfit)
which(lev > 2*p/n)
leveragePlots(reduced.lmfit)
hist(lev, main="Histogram of leverage points", xlab="leverage")
abline(v=2*p/n, lty=2, lwd=2, col="red")

# Detection methods of influential points
# DFFITS
ols_plot_dffits(reduced.lmfit)
dffits(reduced.lmfit)

# Cook's D
ols_plot_cooksd_chart(reduced.lmfit)
cooks.distance(reduced.lmfit)

# DFBETAS
ols_plot_dfbetas(reduced.lmfit)
dfbetas(reduced.lmfit)

### Model Transformation ###
library(EnvStats)

boxcox.summary <- boxcox(reduced.lmfit, optimize=TRUE)
lambda <- boxcox.summary$lambda
lambda
trans.Y <- senic_df$Y^lambda

senic_df <- cbind(senic_df,trans.Y)

## Re-fitting a model using the transformed response variable

boxcox.lmfit <- lm(trans.Y ~ X2 + X7 + X8 + X9 + X1, data=senic_df)
summary(boxcox.lmfit)

boxcox.res <- rstudent(boxcox.lmfit)

boxcox.fitted.y <- fitted(boxcox.lmfit)

# Residual Plots

par(mfrow=c(3,2))
plot(boxcox.res ~ senic_df$X2, xlab="X2", ylab="Residual", main="Residuals vs. X2")
abline(h=0)
plot(boxcox.res ~ senic_df$X7, xlab="X7", ylab="Residual", main="Residuals vs. X7")
abline(h=0)
plot(boxcox.res ~ senic_df$X8, xlab="X8", ylab="Residual", main="Residuals vs. X8")
abline(h=0)
plot(boxcox.res ~ senic_df$X9, xlab="X9", ylab="Residual", main="Residuals vs. X9")
abline(h=0)
plot(boxcox.res ~ senic_df$X1, xlab="X1", ylab="Residual", main="Residuals vs. X1")
abline(h=0)

dev.off()

plot(boxcox.res ~ boxcox.fitted.y, xlab="Fitted value", ylab="Residual", main="Residuals vs. Fitted Values")
abline(h=0)

# Constancy of Error Variances 

bptest(boxcox.lmfit)

# Multicollinearity 

vif(boxcox.lmfit)

# Normality 
qqnorm(boxcox.res);qqline(boxcox.res)
shapiro.test(boxcox.res)

### Final Model ###

final.lmfit <- boxcox.lmfit
summary(final.lmfit)
anova(final.lmfit)


#Required Packages
library(dplyr)
library(caret)
library(class)
library(MASS)
library(ggplot2)
library(boot)
library(corrplot)
library(haven)
library(randomForest)
library(gbm)


my.dir <- "C:/Users/prath/Desktop/Syracuse/Syllabus and Material/SEM 3/PAI793/"

WhiteWine <- read.csv(file = paste0(my.dir,"winequality-whites.csv")
                      ,header = TRUE
                      ,stringsAsFactors = FALSE)

#Checking for NA values and summary stats for white wine
sum(is.na(WhiteWine))
str(WhiteWine)
summary(WhiteWine)
#Dropping the irrelevant column
WhiteWine <- WhiteWine[-1]
hist(WhiteWine$quality)


#Here, let's consider the White Wine dataset as there are more observations and we could build better models.
M <- cor(WhiteWine)
corrplot(M, method = "number")


#Creating a new column classifying quality scores
# 1-5 :Low
# 6-10:High
WhiteWine$quality.bucket <- cut(WhiteWine$quality
                                ,c(1,5,10)
                                ,labels = c("Low", "High"))
#Removing the quality column as our analysis will be only on the bucket
WhiteWine<- WhiteWine[,-12]

#Let's see the distribution of Quality Bucket for alcohol, as it has the higher corellation.
ggplot(data = WhiteWine ,aes(x=quality.bucket, y = alcohol),main ="sasa") + geom_boxplot() +ggtitle("Alcohol Quality Distribution")
ggplot(data = WhiteWine ,aes(x=quality.bucket)) + geom_bar() + ggtitle("Quality Bucket Distribution")

#Diving the dataset into training and testing sets
set.seed(123)
smp_size <- floor(0.75 * nrow(WhiteWine))
train_ind <- sample(seq_len(nrow(WhiteWine)), size = smp_size)
wine_train <- WhiteWine[train_ind, ]
wine_test <- WhiteWine[-train_ind, ]

wine_train_labels <- WhiteWine[train_ind,12]
wine_test_labels <- WhiteWine[-train_ind,12]

#Logit
set.seed(123)
glm.fit2 <- glm(quality.bucket ~., data = wine_train, family = "binomial")
summary(glm.fit2)
glm.pred2 <- predict(glm.fit2,newdata = wine_test,type = "response")
predicted_values <- ifelse(glm.pred2>0.5,"High","Low")
table(predicted_values,wine_test_labels)

#Cross Validation
set.seed(123)
glm.fit <- glm(quality.bucket ~., data = WhiteWine, family = "binomial")
glm.fit
cv.error.1=cv.glm(WhiteWine, glm.fit, K=5)$delta[1]
cv.error.1
cv.error.2=cv.glm(WhiteWine, glm.fit, K=10)$delta[1]
cv.error.2

#LDA
set.seed(123)
library(MASS)
attach(WhiteWine)
lda.fit <- lda(quality.bucket~., data = WhiteWine,subset = train_ind)
lda.fit
#The LDA output indicates that 44.8% of the training observations corresponds to Fairly High wine. Followed by 29.7% for Medium quality wine
#It also provides group means 
plot(lda.fit)
lda.pred <- predict(lda.fit,WhiteWine[-train_ind,])
table(lda.pred$class,wine_test_labels)
1 - mean(lda.pred$class==wine_test_labels) 
#LDA is able to create clear distinct groups for Low and High wines.

#QDA
qda.fit <- qda(quality.bucket~., data = WhiteWine,subset = train_ind)
qda.fit
qda.class <- predict(qda.fit,WhiteWine[-train_ind,])$class
table(qda.class,wine_test_labels)
mean(qda.class==wine_test_labels) 

#KNN
#Here we consider one of the simplest and best-known non-parametric methods, K-nearest neighbors regression (KNN regression)
#Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters.
#A good way to handle this problem is to standardize the data so that all standardize variables are given a mean of zero and a standard deviation of one.
winenew <- scale(WhiteWine[,-13])
winenew <- winenew[,-12]
View(winenew)

set.seed(123)
ctrl <- trainControl(method = "repeatedcv", repeats = 3) #Here we choose repeated cross validation. Repeated 3 means we do everything 3 times for consistency
knnfit <- train(quality.bucket ~., data = WhiteWine, method ="knn", trControl = ctrl, preProcess = c("center","scale"),tuneLength = 20 )
knnfit
plot(knnfit)


#Tree using binary recursive splitting
set.seed(123)
require(tree)
tree.wine<- tree(quality.bucket~., data = wine_train)
summary(tree.wine)
yhat.wine <- predict(tree.wine, newdata = wine_test, type = "class")
table(yhat.wine,wine_test_labels)
print("Mis-classification error",)
(158+170)/1225

#Let's prune this tree using cross validation. How many nodes results in the lowest cross-validation error?
set.seed(123)
cv.tree.wine <- cv.tree(tree.wine,FUN = prune.misclass)
names(cv.tree.wine)
cv.tree.wine
plot(cv.tree.wine$size,cv.tree.wine$dev, type = "b")

#f. Based on the results from cross validation, prune the tree to the smallest number of terminal nodes that reduces the deviance.
prune.wine <- prune.misclass(tree.wine, best=3)
summary(prune.wine)
plot(prune.wine)
text(prune.wine,pretty =0)

tree.pred <- predict(prune.wine,wine_test,type="class")
table(tree.pred,wine_test_labels)
#The testing miscalcullation rate is still the same.


#Bagging
set.seed(123)
bag.wine = randomForest(quality.bucket~.
                       ,data= WhiteWine 
                       ,subset = train_ind  
                       ,mtry= 11, importance =TRUE)
bag.wine

yhat.wine = predict(bag.wine ,newdata = WhiteWine[-train_ind,])
bag_testdf = data.frame(wine_test,yhat.wine)
confusionMatrix(yhat.wine,wine_test$quality.bucket)
#Misclassification error: 199/1225 = 0.1624


#RandomForest
set.seed(123)
rf.wine = randomForest(quality.bucket~.
                        ,data= WhiteWine 
                        ,subset = train_ind  
                        ,mtry= 4, importance =TRUE)
rf.wine

yhat.wine = predict(rf.wine ,newdata = WhiteWine[-train_ind,])
confusionMatrix(yhat.wine,wine_test$quality.bucket)
#M=2 Misclassification error= 190/1225 = 0.1551
#M=3 Misclassification error= 185/1225 = 0.1510
#M=4 Misclassification error= 194/1225 = 0.1583
varImpPlot(bag.wine)
#The first measure is based on how much the accuracy decreases when the variable is excluded.

#The second measure, When a tree is built, the decision about which variable to split at each node uses a calculation of the Gini impurity.
#For each variable, the sum of the Gini decrease across every tree of the forest is accumulated every time that variable is chosen to split a node. 
#The sum is divided by the number of trees in the forest to give an average
#The most important variables are





#Boosting 1##############################################################################################################################

#Converting response to 0's and 1's as we are using bernoulli distribution
set.seed(123)
WhiteWine.gbm <- WhiteWine
WhiteWine.gbm$quality.bucket <- as.integer(WhiteWine.gbm$quality.bucket)
head(WhiteWine.gbm,20)
WhiteWine.gbm <- transform(WhiteWine.gbm, quality.bucket=quality.bucket-1)
str(WhiteWine.gbm)

set.seed(123)
smp_size <- floor(0.75 * nrow(WhiteWine.gbm))
train_ind <- sample(seq_len(nrow(WhiteWine.gbm)), size = smp_size)
gbm_wine_train <- WhiteWine.gbm[train_ind, ]
gbm_wine_test <- WhiteWine.gbm[-train_ind, ]


gbm.model <- gbm(quality.bucket~.
                 ,data = gbm_wine_train
                 ,distribution = "bernoulli"
                 ,n.trees = 10000
                 ,interaction.depth = 2
                 ,shrinkage = 0.01
                 ,cv.folds =5
                 ,verbose = TRUE)
best.iter <- gbm.perf(gbm.model,method="cv") #Check the beat iteration number
best.iter
#Use 9343 trees while actually doing predictions
summary(gbm.model)
#Plots the marginal effect of the selected variables by "integrating" out the other variables.
#How your fetures affect the dependent variable
plot.gbm(gbm.model,1,best.iter)
plot.gbm(gbm.model,2,best.iter)
plot.gbm(gbm.model,3,best.iter)

gbm.pred <- predict.gbm(gbm.model, newdata = gbm_wine_test, n.trees = 9343, type = "response")
gbm.pred.round <- round(gbm.pred)

#Classification Accuracy = 79.34%
1 - sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round))/ length(gbm.pred.round)
#Misclassification rate = 20.65%
sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round))/ length(gbm.pred.round)


#Boosting2 with Shrinkage = 0.001####################################################################################
set.seed  (123)
gbm.model2 <- gbm(quality.bucket~.
                 ,data = gbm_wine_train
                 ,distribution = "bernoulli"
                 ,n.trees = 10000
                 ,interaction.depth = 2
                 ,shrinkage = 0.001
                 ,cv.folds =5
                 ,verbose = TRUE)
best.iter2 <- gbm.perf(gbm.model2,method="cv") #Check the beat iteration number
best.iter2
#Use 10000 trees while actually doing predictions
summary(gbm.model2)


gbm.pred2 <- predict.gbm(gbm.model2, newdata = gbm_wine_test, n.trees = 10000, type = "response")
gbm.pred.round2 <- round(gbm.pred2)

#Classification Accuracy = 78.077%
1 - sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round2))/ length(gbm.pred.round2)
#Mis-classification rate = 21.922%
sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round2))/ length(gbm.pred.round2)


#Boosting3 with depth 3################################################################################
set.seed(123)

gbm.model3 <- gbm(quality.bucket~.
                 ,data = gbm_wine_train
                 ,distribution = "bernoulli"
                 ,n.trees = 10000
                 ,interaction.depth = 3
                 ,shrinkage = 0.01
                 ,cv.folds =5
                 ,verbose = TRUE)
best.iter3 <- gbm.perf(gbm.model3,method="cv") #Check the beat iteration number
best.iter3
#Plots the marginal effect of the selected variables by "integrating" out the other variables.
#How your fetures affect the dependent variable
plot.gbm(gbm.model3,1,best.iter3)
plot.gbm(gbm.model3,2,best.iter3)
plot.gbm(gbm.model3,3,best.iter3)
#Use 6648 trees while actually doing predictions
summary(gbm.model3)

gbm.pred3 <- predict.gbm(gbm.model3, newdata = gbm_wine_test, n.trees = 8982, type = "response")
gbm.pred.round3 <- round(gbm.pred3)

#Classification Accuracy = 80.89%
1 - sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round3))/ length(gbm.pred.round3)
#Misclassification rate = 19.10%
sum(abs(gbm_wine_test$quality.bucket - gbm.pred.round3))/ length(gbm.pred.round3)
###############################################################################################################################
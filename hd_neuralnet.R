
#install.packages("neuralnet")      # Install the neuralnet package if it has not been installed
#install.packages("NeuralNetTools") # An alternative network visualization, predictor importance

# Load required packages
library ("NeuralNetTools")
library("neuralnet") 
library("ggplot2")
library("gridExtra")
library("dplyr")
library("scales")
library("arules")

# Set working directory and read the data
setwd("C:/Users/626854/Desktop/Education/UMGC")

# Read the csv file
hd<-read.csv(file="whas1.csv", head=TRUE, sep=",", as.is=FALSE)
head(hd)
summary(hd)

# Exploratory Data Analysis

# Explore dependent variable distribution
ggplot(hd, aes(x=factor(FSTAT))) + 
  geom_bar(aes(fill=factor(FSTAT))) +
  xlab("Heart Disease") +
  ylab("Count") +
  ggtitle("Heart Disease Survival") +
  scale_fill_discrete(name = "Survived?", 
                      labels = c("Yes", "No")) +
  scale_x_discrete(labels = c('Yes','No'))

# Explore survival rates plotted against various independent variables
grid.arrange(
  ggplot(hd, aes(x = factor(SEX), fill = factor(FSTAT)))+
    geom_bar(position = "fill") +
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No")) +
    scale_x_discrete(labels = c('Male','Female')),
  
  ggplot(hd, aes(x = factor(SHO), fill = factor(FSTAT)))+
    geom_bar(position = "fill")+
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No"))+
    scale_x_discrete(labels = c('No Shock Complications',
                                'Shock Complications')),
  
  
  ggplot(hd, aes(x = factor(CHF), fill = factor(FSTAT)))+
    geom_bar(position = "fill") +
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No"))+
    scale_x_discrete(labels = c('No Heart Failure Complications',
                                'Heart Failure Complications')),
  
  ggplot(hd, aes(x = factor(MIORD), fill = factor(FSTAT)))+
    geom_bar(position = "fill") +
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No"))+
    scale_x_discrete(labels = c('First','Recurrent')),
  
  ggplot(hd, aes(x = discretize(AGE), fill = factor(FSTAT)))+
    geom_bar(position = "fill") +
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No"))+
    scale_x_discrete(labels = c('Young','Middle','Older')),
  
  ggplot(hd, aes(x = discretize(CPK), fill = factor(FSTAT)))+
    geom_bar(position = "fill") +
    scale_fill_discrete(name = "Survived?", 
                        labels = c("Yes", "No"))+
    scale_x_discrete(labels = c('Low','Middle','High')), nrow = 3)


# Data Pre-processing

# Check if the data has missing values
colSums(is.na(hd))
# Scale the numeric variables
hd$AGE<-scale(hd$AGE)
hd$CPK<-scale(hd$CPK)
hd$LENSTAY<-scale(hd$LENSTAY)
hd$LENFOL<-scale(hd$LENFOL)
hd$MITYPE<-scale(hd$MITYPE)
hd$YEAR<-scale(hd$YEAR)
hd$YRGRP<-scale(hd$YRGRP)

# Remove ID
hd$ID<-NULL
# Predicting FSTAT so removing DSTAT since they are both survival variables
hd$DSTAT<-NULL
# Factor dependent variable
hd$FTAT<-factor(hd$FSTAT)

# Divide the data into training and test set
set.seed(1234)
ind <- sample(2, nrow(hd), replace = TRUE, prob = c(0.7, 0.3))
train.data <- hd[ind == 1, ]  #70%
test.data <- hd[ind == 2, ]  #30%

# Use the training data to build the model
# Single hidden layer with 10 nodes
nn <- neuralnet(FSTAT~AGE+SEX+CPK+SHO+CHF+MIORD+MITYPE+YEAR+YRGRP+LENSTAY+
                  LENFOL, data=train.data, hidden=c(10), threshold = 0.2)
# Check the available network properties
names(nn)
nn$model.list # Returns dependent and independent variables
nn$weights #weights for each layer after the last iteration
nn$result.matrix  #the final weights, number of steps, error, threshold
# nn$covariate #are the independent variables from the training data
head(nn$covariate)

# Visualize the network 
plot(nn)
plotnet(nn)
plotnet(nn, circle_col="yellow") #may change node color
# Relative importance for each variable; only for network with 1 hidden layer and one output
garson(nn)  
#Relative importance for each variable; the network may have >=1 hidden layers >=1 output
olden(nn)

# Classification accuracy for training data
mypredict<-neuralnet::compute(nn, nn$covariate)$net.result
mypredict<-apply(mypredict, c(1), round)            # Round the predicted probabilities
table(mypredict, train.data$FSTAT, dnn =c("Predicted", "Actual"))
mean(mypredict==train.data$FSTAT)

# Evaluate the model on a test data
testPred <- neuralnet::compute(nn, test.data[,1:11])$net.result
testPred<-apply(testPred, c(1), round)
testPred
table(testPred, test.data$FSTAT, dnn =c("Predicted", "Actual"))
mean(testPred==test.data$FSTAT)

# Confusion matrix implementation in caret package
require(caret)
require(e1071)
confusionMatrix(table(testPred, test.data$FSTAT), dnn=c("predicted", "actual"))

# neuralnet method documentation and examples
?neuralnet            # View a help page for neuralnet method
?NeuralNetTools       # View a help page for NeuralNetTools package

library("devtools")
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
# may use different color for positive and negative weight connections
plot.nnet(nn, pos.col = "blue", neg.col = "red", circle.col="pink")
#change the circle size
plot.nnet(nn, pos.col = "blue", neg.col = "red", circle.col="pink", circle.cex=3, bord.col="purple")

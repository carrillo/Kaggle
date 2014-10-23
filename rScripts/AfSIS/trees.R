#####################
# Trees for prediction 
#####################

require( "rpart" )
require( "party" )
require( "randomForest" )

#####################
# Example
#####################

# grow tree 
fit <- rpart(Kyphosis ~ Age + Number + Start,
             method="class", data=kyphosis)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postscript plot of tree 
post(fit, file = "c:/tree.ps", 
     title = "Classification Tree for Kyphosis")

##########################
# Load data
##########################
setwd("/Users/carrillo/workspace/Kaggle/resources/AfSIS/")
read.csv("trainingTransformedExpanded.csv",head=T) -> train
train  <- train[ 1:nrow( train )-1, ]
train <- train[,names( train ) != "Ptransformed"]

trainWoId <- train[,names( train ) != "id"]

#fitPart <- rpart( P ~ . , method="anova", data=trainWoId )
#plotcp( fitPart )

fitRandomForest <- randomForest( P ~ . , method="anova", data=trainWoId, ntree=10 )
print( fitRandomForest  )
plot( fitRandomForest )
importance( fitRandomForest )

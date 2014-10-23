######################
# Check results. 
######################


######################
# Functions 
######################

distance  <- function( yPredicted, yObserved ) {
  return( yPredicted - yObserved )
}

meanError <- function( values ) {
  distance  <- distance( values[,1], values[,2] )
  return( sqrt( ( sum( distance^2 ) ) / ( length( distance ) ) ) )
} 

totalError <- function( soc, ph, ca, p, sand ) {
  soc.error <- meanError( soc )
  ph.error <- meanError( ph )
  ca.error <- meanError( ca )
  p.error <- meanError( p )
  sand.error <- meanError( sand )
  
  return( sum( soc.error, ph.error, ca.error, p.error, sand.error)/5 )
}

####################
# Load predictions
####################

soc <- read.table( "~/workspace/Kaggle/output/AfSIS/SOC_XValidation.csv",head=T )
ph <- read.table( "~/workspace/Kaggle/output/AfSIS/pH_XValidation.csv",head=T )
ca <- read.table( "~/workspace/Kaggle/output/AfSIS/Ca_XValidation.csv",head=T )

sand <- read.table( "~/workspace/Kaggle/output/AfSIS/Sand_XValidation.csv",head=T )
meanError( sand )
plot( sand$observation, sand$prediction, cex=0.2 )

p <- read.table( "~/workspace/Kaggle/output/AfSIS/P_XValidation.csv",head=T )
meanError( p )
plot( p$observation, p$prediction, cex=0.2 )


ptransformed <- read.table( "~/workspace/Kaggle/output/AfSIS/Ptransformed_XValidation.csv",head=T )

#####################
# Retransform predictions
#####################

pTransformXOffset <- -0.4217676
pretransformed <- data.frame( 2^( ptransformed ) - pTransformXOffset )
pretransformed[ pretransformed[,1] > max( pretransformed[,2] ), 1]  <- max( pretransformed[,2] )*0.9

####################
# Get errors 
####################

totalError( soc,ph,ca,pretransformed,sand )

#######################
# Test prediction code
#######################

trainingInput <- read.csv( "~/workspace/Kaggle/resources/AfSIS/trainingTransformed.csv",head=T,row.names=c(1) )

CaPrediction <- read.csv( "~/workspace/Kaggle/resources/AfSIS/predict_Ca.csv",head=T,row.names=c(1) )
PPrediction <- read.csv( "~/workspace/Kaggle/resources/AfSIS/predict_P.csv",head=T,row.names=c(1) )

pHPrediction <- read.csv( "~/workspace/Kaggle/resources/AfSIS/predict_pH.csv",head=T,row.names=c(1) )
SOCPrediction <- read.csv( "~/workspace/Kaggle/resources/AfSIS/predict_SOC.csv",head=T,row.names=c(1) )
sandPrediction <- read.csv( "~/workspace/Kaggle/resources/AfSIS/predict_sand.csv",head=T,row.names=c(1) )

predict <- merge( CaPrediction, PPrediction, by="row.names", all.x=T )
rownames( predict ) <- predict$Row.names
predict <- predict[,colnames( predict ) != "Row.names"]

predict <- merge( predict, pHPrediction, by="row.names", all.x=T ) 
rownames( predict ) <- predict$Row.names
predict <- predict[,colnames( predict ) != "Row.names"]

predict <- merge( predict, SOCPrediction, by="row.names", all.x=T ) 
rownames( predict ) <- predict$Row.names
predict <- predict[,colnames( predict ) != "Row.names"]

predict <- merge( predict, sandPrediction, by="row.names", all.x=T ) 
rownames( predict ) <- predict$Row.names
names( predict )[ 1 ] <- "PIDN"

write.csv(x=predict,file="/Users/carrillo/workspace/Kaggle/submissions/predict.csv",quote=F, na="?", row.names=F )

##############################
# Combine predict train with observed  values 
##############################
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predictTrain_Ca.csv",sep=",",head=T ) -> caPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predictTrain_P.csv",sep=",",head=T ) -> pPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predictTrain_pH.csv",sep=",",head=T ) -> pHPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predictTrain_SOC.csv",sep=",",head=T ) -> socPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predictTrain_Sand.csv",sep=",",head=T ) -> sandPredict

predict <- merge( caPredict, pPredict, by="id" )
predict <- merge( predict, pHPredict, by="id" )
predict <- merge( predict, socPredict, by="id" )
predict <- merge( predict, sandPredict, by="id" )

names( predict)  <- paste( names( predict ), "Predicted", sep="_" )
names( predict )[ 1 ] <- "id"

observed <- data.frame( id=rownames( train.output), train.output[, names( train.output ) != "Ptransformed" ] )

newTrainingSet <- merge( predict, observed, by="id" )
write.csv(x=newTrainingSet,file="/Users/carrillo/workspace/Kaggle/resources/AfSIS/callibrationTrain.csv",quote=F, na="?", row.names=F )

read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predict_Ca.csv",sep=",",head=T ) -> caPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predict_P.csv",sep=",",head=T ) -> pPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predict_pH.csv",sep=",",head=T ) -> pHPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predict_SOC.csv",sep=",",head=T ) -> socPredict
read.table( "/Users/carrillo/workspace/Kaggle/resources/AfSIS/predict_Sand.csv",sep=",",head=T ) -> sandPredict

predict <- merge( caPredict, pPredict, by="id" )
predict <- merge( predict, pHPredict, by="id" )
predict <- merge( predict, socPredict, by="id" )
predict <- merge( predict, sandPredict, by="id" )
names( predict)  <- paste( names( predict ), "Predicted", sep="_" )
names( predict )[ 1 ] <- "id"


write.csv(x=predict,file="/Users/carrillo/workspace/Kaggle/resources/AfSIS/callibrationTest.csv",quote=F, na="?", row.names=F )

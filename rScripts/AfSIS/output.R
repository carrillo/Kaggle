############################
# Output tools 
############################

# Merges x and y on row.names, keeps row.names and removes additional row.names column 
mergeOnRowNames <- function( x, y ) {
  out  <- merge( x,y,by="row.names" )
  rownames( out )  <- out$Row.names
  return( out[, names( out ) != "Row.names" ] )
}

writeTestAndTrainToFileFromMergedFeatures <- function( combinedFeatures, train.output, id, train.rowNames, test.rowNames ) {
  # Extract training and test data
  X.train  <- combinedFeatures[ which( rownames( combinedFeatures ) %in% train.rowNames ), ]
  Xy.train <- mergeOnRowNames( X.train, train.output )
  train <- data.frame( PIDN=rownames( Xy.train ), Xy.train )
  
  X.test  <- combinedFeatures[ which( rownames( combinedFeatures ) %in% test.rowNames ), ]
  test <- data.frame( PIDN=rownames( X.test ), X.test )
  
  write.csv(x=train,file=paste( "training",id ,".csv", collapse="", sep="" ),quote=F, na="?", row.names=F )
  write.csv(x=test,file=paste( "test",id ,".csv", collapse="", sep="" ),quote=F, na="?", row.names=F )
}

# Combines spectral and other features
writeTestAndTrainToFile <- function( spectralFeatures, otherFeatures, train.output, id, train.rowNames, test.rowNames ) {
  # Combine collected features with normalized peaks 
  features.new <- mergeOnRowNames( x=otherFeatures,y=spectralFeatures )
  
  # Extract training and test data
  X.train  <- features.new[ which( rownames( features.new ) %in% train.rowNames ), ]
  Xy.train <- mergeOnRowNames( X.train, train.output )
  train <- data.frame( PIDN=rownames( Xy.train ), Xy.train )
  
  X.test  <- features.new[ which( rownames( features.new ) %in% test.rowNames ), ]
  test <- data.frame( PIDN=rownames( X.test ), X.test )
  
  write.csv(x=train,file=paste( "training",id ,".csv", collapse="", sep="" ),quote=F, na="?", row.names=F )
  write.csv(x=test,file=paste( "test",id ,".csv", collapse="", sep="" ),quote=F, na="?", row.names=F )
}
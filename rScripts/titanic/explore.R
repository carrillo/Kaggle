###########################
# Explore titanic data 
###########################

write  <- FALSE

###########################
# Load libraries and set working directory 
###########################

library( ggplot2 )
library( reshape )
setwd("~/workspace/Kaggle/resources/titanic/")

###########################
# Clean data for R 
# 1. Read data 
# 2. Replace '?' used in Weka for NAs 
# 3. Make sure that features have the right class
###########################
d <- read.csv("trainClean.csv",head=T) 
d[ d == "?" ] <- NA 
d$Survived  <- as.factor( d$Survived ) # 
d$Pclass  <- as.factor( d$Pclass )
d$Age <- as.numeric( as.character( d$Age ) ) 
d$TicketNr <- as.numeric( as.character( d$TicketNr ) ) 
d$CabinCount <- as.numeric( as.character( d$CabinCount ) ) 
d$CabinNr <- as.numeric( as.character( d$CabinNr ) ) 

###########################
# What is the prior propability of survival? 
###########################

survived.fraction <- sum( as.numeric( as.character( d$Survived ) ) ) / length( d$Survived )

###########################
# NOMINAL FEATURES
# 1. Subset data for nominal features
# 2. Calculate fraction survived for each class within the nominal features 
###########################
d.nominal  <- d[,which( sapply(d, is.factor ) ) ]
d.nominal.features  <- names( d.nominal )[ which( names( d.nominal ) != "Survived" ) ]
d.nominal$Survived  <- as.numeric( as.character( d.nominal$Survived ) )

fraction.survived <- function( featureName, d.nominal ) {
  index  <- which( names( d.nominal ) == featureName ) 
  df <- data.frame( value=d.nominal[,index], sex=d.nominal$Sex, Survived=d.nominal$Survived )
  df  <- df[ !is.na( df$value ), ]
  
  fraction.survived.perLevel <- function( level ) {
    df.levelsub <- df[ df$value == level, ]
    n <- length( df.levelsub$value )
    survivedMale <- sum( df.levelsub[ df.levelsub$sex == "male", ]$Survived )
    survivedFemale <- sum( df.levelsub[ df.levelsub$sex == "female", ]$Survived )
    out  <- data.frame( feature=featureName, value=level, fractionSurvived=mean( df.levelsub$Survived ), fraction.survived.female= survivedFemale/n, fractionSurvived.male = survivedMale/n, n=n )
    return(  out ) 
  }
  
  out  <- data.frame() 
  for( i in levels( df$value ) ) {
    if( i != "?" ) { out <- rbind( out, fraction.survived.perLevel( i ) )  }
  }
  return( out )
}

# Get fraction survival for nominal features, consider only categories with > 10 samples 
nominal.fractionSurvived  <- data.frame() 
for( i in d.nominal.features ) {
  nominal.fractionSurvived <- rbind( nominal.fractionSurvived, fraction.survived( i, d.nominal ) )
}
nominal.fractionSurvived <- nominal.fractionSurvived[ nominal.fractionSurvived$n >= 10, ] 

quartz()
p <- ggplot( nominal.fractionSurvived, aes( x = factor( value ), y = fractionSurvived  )  )
p <- p + geom_bar( stat = "identity",fill="white", colour="darkgreen" )
#p <- p + geom_hline( survived.fraction )
p <- p + facet_wrap( ~ feature, scale="free",  )
#p  <- p + scale_x_continuous( limits=c(0,1), name="scaled value" )
p  <- p + scale_y_continuous( limits=c(0,1), name="fraction survived" )
show( p )

###########################
# NUMERICAL FEATURES
# Test which features are statistically difference between classes. 
# 1. Subset data for numeric features + class feature (Survived)
# 2. Perform 2-sample Kolmogorov-Smirnov test.    
###########################
d.numeric  <- d[,which( sapply(d, is.numeric) ) ]
d.numeric.features  <- names( d.numeric )
d.numeric$Survived  <- d$Survived

kolmogorov.test <- function( featureName, d.numeric ) {
  index  <- which( names( d.numeric ) == featureName ) 
  df <- data.frame( value=d.numeric[,index], Survived=d.numeric$Survived )
  
  notSurvived <- df[ df$Survived == 0 , 1 ]
  survived <- df[ df$Survived == 1 , 1 ]
  test  <- ks.test( survived, notSurvived )
  return( test$p.value  )
}

pValues  <- sapply( d.numeric.features, function ( x ) kolmogorov.test( x, d.numeric ) )

###########################
# Plot data using empirical cumulative distributions
# 1. Scale all numerical values between 0 and 1 
# 2. Reformat the data such that each row contains the numeric value, the feature and the survival
# 2. Plot   
###########################
d.numeric  <- d[,which( sapply(d, is.numeric) ) ]
d.numeric.scaled  <- data.frame( apply( d.numeric, MARGIN=2, function( x ) ( x - min( x, na.rm=T ) ) / diff( range( x, na.rm=T ) ) ) )
d.numeric.scaled$Survived  <- d$Survived

d.numeric.scaled.melt <- ( melt( d.numeric.scaled ) )
d.numeric.scaled.melt <- d.numeric.scaled.melt[ !is.na( d.numeric.scaled.melt$value ), ]

quartz()
p <- ggplot( d.numeric.scaled.melt, aes( value, colour = Survived ) )
p <- p + stat_ecdf() 
p <- p + facet_wrap(~ variable, ncol=4 )
p  <- p + scale_x_continuous( limits=c(0,1), name="scaled value" )
p  <- p + scale_y_continuous( limits=c(0,1), name="cumulative probability" )

if( write ) {
  fileName <- "exploration/numericFeaturesECDF.png"
  png( fileName )
  show( p )
  dev.off()  
} else {
  show( p )
}

###########################
# NOMINAL FEATURES
# 1. Subset data for nominal features
# 2. Calculate fraction survived for each class within the nominal features 
###########################
d.nominal  <- d[,which( sapply(d, is.factor ) ) ]
d.nominal.features  <- names( d.nominal )[ which( names( d.nominal ) != "Survived" ) ]
d.nominal$Survived  <- as.numeric( as.character( d.nominal$Survived ) )

fraction.survived <- function( featureName, d.nominal ) {
  index  <- which( names( d.nominal ) == featureName ) 
  df <- data.frame( value=d.nominal[,index], sex=d.nominal$Sex, Survived=d.nominal$Survived )
  df  <- df[ !is.na( df$value ), ]
  
  fraction.survived.perLevel <- function( level ) {
    df.levelsub <- df[ df$value == level, ]
    n <- length( df.levelsub$value )
    survivedMale <- sum( df.levelsub[ df.levelsub$sex == "male", ]$Survived )
    survivedFemale <- sum( df.levelsub[ df.levelsub$sex == "female", ]$Survived )
    out  <- data.frame( feature=featureName, value=level, fractionSurvived=mean( df.levelsub$Survived ), fraction.survived.female= survivedFemale/n, fractionSurvived.male = survivedMale/n, n=n )
    return(  out ) 
  }
  
  out  <- data.frame() 
  for( i in levels( df$value ) ) {
    if( i != "?" ) { out <- rbind( out, fraction.survived.perLevel( i ) )  }
  }
  return( out )
}

# Get fraction survival for nominal features, consider only categories with > 10 samples 
nominal.fractionSurvived  <- data.frame() 
for( i in d.nominal.features ) {
  nominal.fractionSurvived <- rbind( nominal.fractionSurvived, fraction.survived( i, d.nominal ) )
}
nominal.fractionSurvived <- nominal.fractionSurvived[ nominal.fractionSurvived$n >= 10, ] 

quartz()
p <- ggplot( nominal.fractionSurvived, aes( x = factor( value ), y = fractionSurvived  )  )
p <- p + geom_bar( stat = "identity",fill="white", colour="darkgreen" )
p <- p + facet_wrap( ~ feature, scale="free",  )
#p  <- p + scale_x_continuous( limits=c(0,1), name="scaled value" )
p  <- p + scale_y_continuous( limits=c(0,1), name="fraction survived" )
show( p )


##################################
# Characteristic decay in absorbance spectra
##################################

# Determine charactersitic decay by fitting a linear function in linear-log space. return slope as feature
quantifyLinearDecreaseInLinLog <- function( Y ) {
  
  getSlope <- function( x, y ) {
    model <- lm( y ~ x )
    return( coefficients( model )[[ 2 ]] )
  }
  
  x  <- log10( as.numeric( colnames( Y ) ) )
  m <- data.frame( decay=apply( Y, MARGIN=1, getSlope, x = x ) )
  
  return( m )
}

# Substract decay in lin log space. 
makeLinLogAndSubstractDecay <- function( Y ) {
  getCorrectedY <- function( x, y ) {
    model  <- lm( y ~ x ) 
    yPredict  <- predict( model ) 
    return( y - yPredict )
  }
  
  x  <- log10( as.numeric( colnames( Y ) ) )
  m  <- data.frame( t( apply( Y, MARGIN=1, getCorrectedY, x=x ) ) )
  colnames( m )  <- colnames( Y )
  return( m )
}


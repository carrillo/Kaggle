#######################
# Scale spectra 
#######################

spectra.scaleBetweenMinAndMedian  <- function( Y ) {
  
  range01 <- function( x ) {
    ( x - min( x ) ) / ( median( x ) - min( x ) )
  }
  
  m <- data.frame( t( apply( X=Y, MARGIN=1, range01 ) ) )
  
  colnames( m ) <- colnames( Y )
  
  return( m )
}

spectra.scaleBetweenMinAndMax  <- function( Y ) {
  
  range01 <- function( x ) {
    ( x - min( x ) ) / ( max( x ) - min( x ) )
  }
  
  m <- data.frame( t( apply( X=Y, MARGIN=1, range01 ) ) )
  
  colnames( m ) <- colnames( Y )
  
  return( m )
}

spectra.scaleBetweenMinAndMean  <- function( Y ) {
  
  range01 <- function( x ) {
    ( x - min( x ) ) / ( mean( x ) - min( x ) )
  }
  
  m <- data.frame( t( apply( X=Y, MARGIN=1, range01 ) ) )
  
  colnames( m ) <- colnames( Y )
  
  return( m )
}




###########################
# Smooth spectra by gaussian kernel method
###########################

spectra.LogX <- function( Y ) {
  Y1 <- Y
  colnames( Y1 ) <- log10( as.numeric( colnames( Y ) ) )
  return( Y1 )
}

# smooth splectra with a gaussian kernel with defined bandwith 
spectra.smooth  <- function( Y, bandwith=0.01 ) {
  
  smoothSerie <- function( x, y ) {
    return( ksmooth( x=x, y=y, kernel="normal", bandwidth=bandwith ) )     
  }
  
  getX <- function( x, y ) {
    s <- smoothSerie( x, y )
    return( s$x )
  }
  
  getY <- function( x, y ) {
    s <- smoothSerie( x, y )
    return( s$y )
  }
  
  x <- as.numeric( colnames( Y ) ) 
  
  m <- data.frame( t( apply(X=Y, MARGIN=1, getY, x=x ) ) )
  names( m ) <- getX(  x, Y[1,] )
  
  return( m )
}

#############################
# Visualize principal components 
#############################

require( "ggplot2" )

pc.getVarianceData <- function( pc, maxPC=NA ) {
  if( is.na( maxPC ) ) {
    maxPC <- length( pc$sdev )
  }
  
  df<- data.frame( variance=(pc$sdev[1:maxPC])^2, pc=c(1:length(pc$sdev[1:maxPC]) ) )
  df$variance <- df$variance/sum( df$variance ) 
  
  return( df )
}

pc.plotExplainedVariation <- function( pc, maxPC=NA, xlog=FALSE, ylog=FALSE ) {
  df <- pc.getVarianceData( pc, maxPC )
  
  p <- ggplot( df, aes(x=pc,y=variance) )
  p <- p + geom_point()
  p <- p + geom_line()
  
  if( ylog ) {
    p <- p + scale_y_log10( "Fraction of total variance" )
    p <- p + annotation_logticks(sides="l")
  } else {
    p <- p + scale_y_continuous( "Fraction of total variance" )
  }
  if( xlog ) {
    p <- p + scale_x_log10( "Principal component" )
    p <- p + annotation_logticks(sides="b")
  } else {
    p <- p + scale_x_continuous( "Principal component" )
  }
  
  return( p )
}
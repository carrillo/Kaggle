###############################
# Plotting spectra
###############################

# Returns the x values from the frequencies given in column names. 
getX <- function( Y ) {
  return( as.numeric( colnames( Y ) ) )
}

getY <- function( Y, rowIndex ) {
  return( as.numeric( Y[ rowIndex, ] ) )
}

# Plot one or more spectra specified by rowIndex 
spectrum.plot <- function( Y, rowIndex, title=NA ) {
  ids <- rownames( Y )
  df <- data.frame()
  for( i in 1:length( rowIndex ) ) {
    x <- getX( Y )
    y <- getY( Y, i )
    id <- ids[ i ]
    df  <- rbind( df, data.frame( x=x, y=y, id=id ) )
  }
  
  p <- ggplot( df, aes( x, y, colour=id ) )
  if( !is.na( title ) ) {
    p <- p + labs(title=title)
  }
  
  p <- p + geom_line()
  p <- p + scale_x_continuous("Wavenumber (1/cm)")
  p <- p + scale_y_continuous("Intensity")
  p <- p + theme(legend.justification=c(1,1), legend.position=c(1,1))
  
  return( p )
}


# Plot spectra as heatmap 
spectra.heatmap <- function( Y ) {
  heatmap.2( as.matrix( Y ),
             Colv="NA", 
             dendrogram='row', 
             trace="none", 
             labRow=F, 
             labCol=F ) 
}
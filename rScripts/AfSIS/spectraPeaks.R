###############################
# Detect peaks in spectra
###############################

# Get peaks for given spectrum 
spectrum.getPeakXY <- function( x, y ) {
  peaks  <- findpeaks( as.numeric( y ),nups=10, ndowns=10 )
  return( data.frame( x=x[ peaks[ ,2 ] ], y=peaks[,1]) )
}

# Returns all peak positions over all spectra in Y 
spectra.getAllPeaks <- function( Y ) {
  x <- as.numeric( colnames( Y ) )
  
  # Get all peaks 
  peaks.list <- apply( X=Y, MARGIN=1, spectrum.getPeakXY, x=x )
  
  peaks <- data.frame() 
  for( i in 1:length( peaks.list ) ) {
    df <- data.frame( peaks.list[ i ], label=names( peaks.list[ i ] ) )
    names( df ) <- c("x","y","label")
    peaks <- rbind( peaks, df )
  }
  
  return( peaks )
}

# Returns peak regions defined as a range of x where a minimal number x of all spectra show peaks 
spectra.getPeakRegions <- function( Y, histBreaks=1000, minSpectraWithPeak=10, plot ) {
  
  # find connected regions in the histogram over a given threshold -> define peak xranges. 
  connectedRegionsOverThreshold <- function( histogram, threshold ) {
    inRegion <- F
    
    start <- -1
    end <- -1 
    
    df <- data.frame()
    
    for( i in 1:length( histogram$mids ) ) {
      
      y <- histogram$counts[ i ]
      
      if( y >= threshold ) {
        if( !inRegion ) {
          start <- i - 1  
          inRegion  <- TRUE
        } 
      } else {
        if( inRegion ) {
          end <- ( i )
          inRegion  <- F
          
          df <- rbind( df, data.frame( start=start, end=end ) )
        } 
      }
    }
    
    if( inRegion ) {
      end <- i 
      df <- rbind( df, data.frame( start=start, end=end ) )  
    }
    
    
    #df  <- df[ ( df$end - df$start ) != 0, ] 
    df <- data.frame( start=histogram$mids[ df$start ], end=histogram$mids[ df$end ] )
    return( df )
  }
  
  # Get all peaks in data 
  peaks  <- spectra.getAllPeaks( Y )
  
  # Count number of per x-bin over all spectra 
  peakHistogram <- hist( peaks$x, breaks=histBreaks )
 
  # Find connected x values > threshold in histogram 
  peakRegions <- connectedRegionsOverThreshold( histogram=peakHistogram, threshold=minSpectraWithPeak )
  
  if( plot ) {
    quartz()
    plot( peakHistogram$mids, peakHistogram$counts, type='l' )
    arrows( peakRegions$start, minSpectraWithPeak, peakRegions$end, minSpectraWithPeak, code=3, angle=90, col='red' )  
  }
  
  return( peakRegions )
}

# Extract x,y of maxima within defined peak region for spectrum 
spectrum.getPeakXYFeatures <- function( x, y, peakRegions, print=F ) {

  getXY <- function( x, y, peakRegionRow ) {
    start <- peakRegionRow[[ 1 ]]
    end <- peakRegionRow[[ 2 ]]
    id <- mean( c( start, end) )
    
    indexSubrange  <- which( x >= start & x <=end )
    ySub <- y[ indexSubrange ]
    
    yMax <- max( ySub )
    xMax <- x[ which.max( ySub )[[ 1 ]] + indexSubrange[ 1 ] - 1 ]
    
    
    return( data.frame(xMax=xMax,yMax=yMax,row.names=id) )
  }
  
  peakXY <- data.frame() 
  for( i in 1:nrow( peakRegions ) ) {
    peakXY <- rbind( peakXY, getXY( x=x, y=y, peakRegionRow=peakRegions[i,] ) )  
  }
  
  if( print ) {
    quartz() 
    plot( x, y, type='l' )
    points( df$xMax, df$yMax, col='red')
  }
  
  t( peakXY )
  #featureNames <- 
  
  peaksX  <- data.frame( value=peakXY$xMax, row.names=paste( rownames( peakXY ),rep("PeakPosX",nrow( peakXY ) ), sep="_" ) )
  peaksY  <- data.frame( value=peakXY$yMax, row.names=paste( rownames( peakXY ),rep("PeakPosY",nrow( peakXY ) ), sep="_" ) )
  
  peakFeatures <- t( rbind( peaksX, peaksY ) )
  return( peakFeatures )
}

spectra.getPeakFeatures <- function( Y, peakRegions ) {

  # Extract table with peak features
  x <- as.numeric( colnames( Y ) )
  peaksFeatureXY <- t( apply( X=Y, MARGIN=1, spectrum.getPeakXYFeatures, x=x, peakRegions=peakRegions ) )
  
  # Name features accordingly: peakId_PeakPosX and peakId_PeakPosY
  peakIds <- as.character( apply( peakRegions, MARGIN=1, mean ) )
  colnames( peaksFeatureXY )  <- c( paste( peakIds, rep("PeakPosX"), sep="_" ), paste( peakIds, rep("PeakPosY"), sep="_" ) )
  
  return( peaksFeatureXY )
}


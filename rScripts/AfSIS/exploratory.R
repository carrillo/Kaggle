##################################
# Feature extraction for AfSIS Kaggle competition
# 
# 1. Raw data preparation
# 2. Extract characteristic decay
# 3. Principal component analysis
#
##################################

##################################
# Dependencies 
##################################
require( "gplots" )
require( "ggplot2" )
require( "pracma" )

source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/spectraPlot.R")
source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/characteristicDecay.R")
#source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/smoothSpectra.R")
#source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/scaleSpectra.R")
#source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/spectraPeaks.R")
source("/Users/carrillo/workspace/Kaggle/rScripts/AfSIS/output.R")

plot <- FALSE
##################################
# Raw data preparation
# 1. Load training and test files
# 2. Extract spectra and other data
##################################
setwd("/Users/carrillo/workspace/Kaggle/resources/AfSIS/")
train <- read.table("training.csv",head=T,sep=",",row.names=1)
test <- read.table("sorted_test.csv",head=T,sep=",",row.names=1)

# Define header section for spectral and other input
absorbanceFreqStartIndex  <- 3578
absorbanceFreqEndIndex  <- 1
otherStartIndex <- 3579
otherEndIndex <- 3594 
outputStartIndex <- 3595
outputEndIndex <- 3599

# Test if headers are the same for test and train data
# which( names( train[absorbanceFreqStartIndex:absorbanceFreqEndIndex] ) != names( test[absorbanceFreqStartIndex:absorbanceFreqEndIndex] ) )
# which( names( train[otherStartIndex:otherEndIndex] ) == names( test[otherStartIndex:otherEndIndex] ) )

# Extract spectrum data 
train.spectra  <- train[ absorbanceFreqStartIndex : absorbanceFreqEndIndex ]
test.spectra  <- test[ absorbanceFreqStartIndex : absorbanceFreqEndIndex ]

frequencies <- as.numeric( sub("m",replacement="", names( train.spectra ) ) )
names( train.spectra ) <- frequencies
names( test.spectra ) <- frequencies
rm( absorbanceFreqEndIndex, absorbanceFreqStartIndex, frequencies )

# Plot raw spectra
if( plot ) {
  quartz()
  spectrum.plot( train.spectra, rowIndex=seq(1:10) )
  spectrum.plot( test.spectra, rowIndex=seq(1:10) )  
}

# Extract non-spectral features 
train.inputOther <- train[ otherStartIndex : otherEndIndex ]
test.inputOther <- test[ otherStartIndex : otherEndIndex ]
rm( otherStartIndex, otherEndIndex )

# Extract output features
train.output <- train[ outputStartIndex : outputEndIndex ]
rm( outputStartIndex, outputEndIndex )

# Remove raw input files. 
rm( train, test )

# Plot output correlations 
if( plot ) {
  quartz()
  pairs(train.output, main="Output Scatterplot Matrix", cex=0.1 )
}

# Merge test and train spectra for normalization, keep row information
combined.spectra  <- rbind( train.spectra, test.spectra )
combined.inputOther <- rbind( train.inputOther, test.inputOther )


feature.collect <- combined.inputOther

# Make Depth numeric: Subsoil = 0, Topsoil = 1  
feature.collect$Depth  <- as.character( feature.collect$Depth )
feature.collect$Depth[ feature.collect$Depth == "Subsoil" ]  <- 0
feature.collect$Depth[ feature.collect$Depth == "Topsoil" ]  <- 1
feature.collect$Depth  <- as.numeric( feature.collect$Depth )

################################
# Collect summary statistics of original data
################################
feature.collect$originalMean <- apply( combined.spectra, MARGIN=1, mean )
feature.collect$originalMedian <- apply( combined.spectra, MARGIN=1, median )
feature.collect$originalMax <- apply( combined.spectra, MARGIN=1, max )
feature.collect$originalMin <- apply( combined.spectra, MARGIN=1, min )
feature.collect$originalRange <- feature.collect$originalMax - feature.collect$originalMin
feature.collect$sd  <- apply( combined.spectra, MARGIN=1, sd )

feature.collect$originalQuantile10  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.1) )
feature.collect$originalQuantile20  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.2) )
feature.collect$originalQuantile30  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.3) )
feature.collect$originalQuantile40  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.4) )
feature.collect$originalQuantile60  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.6) )
feature.collect$originalQuantile70  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.7) )
feature.collect$originalQuantile80  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.8) )
feature.collect$originalQuantile90  <- apply( combined.spectra, MARGIN=1, quantile, probs=c(0.9) )

################################
# Determine the characteristic decay for each sample. 
# 1. Extract characteristic slope and save as feature. 
# 2. Transfrom to linear-log space
# 3. Substract decay 
# 4. Collect summary statistics in featureset
################################

# Extract slope
decay <-  quantifyLinearDecreaseInLinLog( Y=combined.spectra )
feature.collect <- data.frame( feature.collect, decay )
rm( decay )

# Transform and Substract slope
combined.spectra.linLog.decayCorrect <- makeLinLogAndSubstractDecay( combined.spectra )

# Collect summary statistics 
feature.collect$mean <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, mean )
feature.collect$quantile10  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.1) )
feature.collect$quantile20  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.2) )
feature.collect$quantile30  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.3) )
feature.collect$quantile40  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.4) )
feature.collect$median <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, median )
feature.collect$quantile60  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.6) )
feature.collect$quantile70  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.7) )
feature.collect$quantile80  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.8) )
feature.collect$quantile90  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, quantile, probs=c(0.9) )
feature.collect$sd  <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, sd )
feature.collect$max <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, max )
feature.collect$min <- apply( combined.spectra.linLog.decayCorrect, MARGIN=1, min )
feature.collect$range <- feature.collect$max - feature.collect$min

ggsave( spectrum.plot( combined.spectra, c(1:10) ), filename="~/Projects/Homepage/content/kaggleAfricanSoil/plots/raw.png", width=7, height=7)
ggsave( spectrum.plot( combined.spectra.linLog.decayCorrect, c(1:10) ), filename="~/Projects/Homepage/content/kaggleAfricanSoil/plots/decaySubtracted.png", width=7, height=7)

cp <- colorRampPalette( c("red","yellow","blue"), interpolate ="linear", space="Lab", bias=1 )
quartz()
png( "~/Projects/Homepage/content/kaggleAfricanSoil/plots/heatmap.png" )
heatmap.2( as.matrix( combined.spectra.linLog.decayCorrect ), Colv="NA", dendrogram='row', col=cp(100), trace="none", labRow=F, labCol=F )
dev.off()
################################
# Reduce dimensionality by PCA 
################################
pcBoth <- prcomp(  mergeOnRowNames(x=combined.spectra.linLog.decayCorrect,y=feature.collect)  )
writeTestAndTrainToFileFromMergedFeatures( combinedFeatures=data.frame( pcBoth$x ), train.output=train.output, 
                                           id="DecaySubtractedPrincipalComponents", train.rowNames=rownames( train.spectra ), test.rowNames=rownames( test.spectra ) )


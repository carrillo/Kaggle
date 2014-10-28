#############################
# Plot error 
#############################
require( "ggplot2" ); require(scales);   

errors <- read.csv( "~/Projects/Homepage/content/kaggleAfricanSoil/plots/errorsVsPC.txt",head=T, colClasses=c("numeric","factor",rep("numeric",5) )  )
errors$meanError  <- apply( errors, MARGIN=1, function( x ) { mean( as.numeric( x[3:7] ) ) } )

# The palette with grey:
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

p <- ggplot( errors, aes( x=PC, y=meanError, color=Data_Set ) )
p <- p + geom_point()
p <- p + geom_line()
p <- p + scale_colour_manual(values=cbPalette)
p <- p + scale_x_continuous("Principal component count (Log2)",trans=log2_trans(),breaks=2^c(1:9))
p <- p + scale_y_continuous("Cross validation error (RMSE)",limits=c(0,max(errors$meanError)))
p <- p + theme(legend.justification=c(1,1), legend.position=c(1,1))

show( p )

ggsave( p, filename='/Users/carrillo/Projects/Homepage/content/kaggleAfricanSoil/plots/errorsVsPC.png', width=7, height=7)


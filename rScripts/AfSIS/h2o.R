###################
# Prediction using H20 
# http://0xdata.com/blog/2014/09/r-h2o-domino/
################### 

## Specify H2O version here
h2o_ver <- "1511"

## Install H2O
local({r <- getOption("repos"); r["CRAN"] <- "http://cran.us.r-project.org"; options(repos = r)})
txt_repo <- (c(paste0(paste0("http://s3.amazonaws.com/h2o-release/h2o/master/", h2o_ver),"/R"), getOption("repos")))
install.packages("h2o", repos = txt_repo, quiet = TRUE)

# Initiate and connect to a cluster
library(h2o)
localH2O <- h2o.init(max_mem_size = '6g')

# Load data 
train_hex <- h2o.importFile(localH2O, "/Users/carrillo//workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedSmooth5e-04.csv")
test_hex <- h2o.importFile(localH2O, "/Users/carrillo//workspace/Kaggle/resources/AfSIS/testDecaySubtractedSmooth5e-04.csv")

# Load submission template 
raw_sub  <- read.csv("/Users/carrillo//workspace/Kaggle/resources/AfSIS/sample_submission.csv")

## Split the dataset into 80:20 for training and validation
train_hex_split <- h2o.splitFrame(train_hex, ratios = 0.8, shuffle = TRUE)

## One Variable at at Time
ls_label <- c("Ca", "P", "pH", "SOC", "Sand")

for (n_label in 1:5) {
  
  ## Display
  cat("\n\nNow training a DNN model for", ls_label[n_label], "...\n")
  
  ## Train a 50-node, three-hidden-layer Deep Neural Networks for 100 epochs
  model <- h2o.deeplearning(x = 2:3595,
                            y = (3595 + n_label),
                            data = train_hex_split[[1]],
                            validation = train_hex_split[[2]],
                            activation = "Rectifier",
                            hidden = c(50, 50, 50),
                            epochs = 100,
                            classification = FALSE,
                            balance_classes = FALSE)
  
  ## Print the Model Summary
  print(model)
  
  ## Use the model for prediction and store the results in submission template
  raw_sub[, (n_label + 1)] <- as.matrix(h2o.predict(model, test_hex))
  
}

write.csv(raw_sub, file = "/Users/carrillo//workspace/Kaggle/output/AfSIS/predictionH20.csv", row.names = FALSE)

##################
# Second script 
# https://github.com/0xdata/h2o/blob/master/R/examples/Kaggle/MeetupKaggleAfricaSoil.R
##################
#install.packages("/Users/carrillo/Programs/h2o-2.7.0.1538/R/h2o_2.7.0.1538.tar.gz",repos = NULL, type = "source")
install.packages("/Users/carrillo/Programs/h2o-2.7.0.1511/R/h2o_2.7.0.1511.tar.gz",repos = NULL, type = "source")
#install.packages("/Users/carrillo/Programs/h2o-2.8.0.1/R/h2o_2.8.0.1.tar.gz",repos = NULL, type = "source")
#install.packages("/Users/carrillo/Programs/h2o-2.6.1.5/R/h2o_2.6.1.5.tar.gz",repos = NULL, type = "source")
require( h2o )
demo( h2o.glm )

# h2o.shutdown( localH2O, prompt=TRUE )

remoteH2O <- h2o.init(ip = '172.29.13.214', port = 54321, max_mem_size = '60g')
#localH2O <- h2o.init(max_mem_size = '8g')
# Load data 

#train_hex <- h2o.importFile(localH2O, "/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedSmooth5e-04.csv")
#test_hex <- h2o.importFile(localH2O, "/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtractedSmooth5e-04.csv")

train_hex <- h2o.importFile(remoteH2O, "/home/carrillo/h2o/data/trainingDecaySubtracted128PcFromAllFeaturesNew.csv")
test_hex <- h2o.importFile(remoteH2O, "/home/carrillo/h2o/data/testDecaySubtracted128PcFromAllFeaturesNew_DepthNumeric.csv")


# Group variables
vars <- colnames(train_hex)
#spectra <- vars[seq(30,length( vars )-5,by=1)]
#extra <- vars[2:29]
targets <- vars[ ( length( vars ) - 4 ) : length( vars ) ]
#predictors <- c(spectra, extra)
predictors <- vars[ 2:(length( vars )-5)]


# LB score of 0.439
# ensemble_size <- 20
# n_fold = 20
ensemble_size <- 20
n_fold = 20

# Scoring helpers
MSEs <- matrix(0, nrow = 1, ncol = length(targets))
RMSEs <- matrix(0, nrow = 1, ncol = length(targets))
CMRMSE = 0

setwd( "/Users/carrillo/workspace/Kaggle/output/AfSIS/h2o/128pcAllNewNotMatched/" )

# Main loop over regression targets
for (resp in 1:length(targets)) {
  cat("\n\nNow training and cross-validating a DL model for", targets[resp], "...\n")
  
  # Run grid search with n-fold cross-validation
  cvmodel <-
    h2o.deeplearning(x = predictors,
                     y = targets[resp],
                     data = train_hex,
                     nfolds = n_fold,
                     classification = F,
                     activation="RectifierWithDropout",
                     hidden = c(100,100),
                     hidden_dropout_ratios = c(0.0,0.0),
                     input_dropout_ratio = 0,
                     epochs = 100,
                     l1 = c(0,1e-5), 
                     l2 = c(0,1e-5), 
                     rho = 0.99, 
                     epsilon = 1e-8, 
                     train_samples_per_iteration = -2
    )
  
  ## Collect cross-validation error
  MSE <- cvmodel@sumtable[[1]]$prediction_error   #If cvmodel is a grid search model
  #MSE <- cvmodel@model$valid_sqr_error            #If cvmodel is not a grid search model
  RMSE <- sqrt(MSE)
  CMRMSE <- CMRMSE + RMSE #column-mean-RMSE
  MSEs[resp] <- MSE
  RMSEs[resp] <- RMSE
  cat("\nCross-validated MSEs so far:", MSEs)
  cat("\nCross-validated RMSEs so far:", RMSEs)
  cat("\nCross-validated CMRMSE so far:", CMRMSE/resp)
  
  cat("\n\nTaking parameters from grid search winner for", targets[resp], "...\n")
  p <- cvmodel@sumtable[[1]]  #If cvmodel is a grid search model
  #p <- cvmodel@model$params   #If cvmodel is not a grid search model
  
  ## Build an ensemble model on full training data
  for (n in 1:ensemble_size) {
    cat("\n\nBuilding ensemble model", n, "of", ensemble_size, "for", targets[resp], "...\n")
    model <-
      h2o.deeplearning(x = predictors,
                       y = targets[resp],
                       key = paste0(targets[resp], "_cv_ensemble_", n, "_of_", ensemble_size),
                       data = train_hex, 
                       classification = F,
                       activation = p$activation,
                       hidden = p$hidden, 
                       hidden_dropout_ratios = p$hidden_dropout_ratios,
                       input_dropout_ratio = p$input_dropout_ratio,
                       epochs = p$epochs,
                       l1 = p$l1,
                       l2 = p$l2,
                       rho = p$rho,
                       epsilon = p$epsilon,
                       train_samples_per_iteration = p$train_samples_per_iteration)
    
    ## Aggregate ensemble model predictions
    test_preds <- h2o.predict(model, test_hex)
    
    if (n == 1) {
      test_preds_blend <- test_preds
    } else {
      test_preds_blend <- cbind(test_preds_blend, test_preds[,1])
    }
  }
  
  ## Now create submission
  cat (paste0("\n Number of ensemble models: ", ncol(test_preds_blend)))
  ensemble_average <- matrix("ensemble_average", nrow = nrow(test_preds_blend), ncol = 1)
  ensemble_average <- rowMeans(as.data.frame(test_preds_blend)) # Simple ensemble average, consider blending/stacking
  ensemble_average <- as.data.frame(ensemble_average)
  
  colnames(ensemble_average)[1] <- targets[resp]
  
  currentSubmission <- cbind(as.data.frame(test_hex[,1]), ensemble_average)
  write.csv(currentSubmission, file =paste( "prediction", targets[ resp ], ".csv", sep="", collapse="" ), quote = F, row.names=F)
  
  if (resp == 1) {
    final_submission <- cbind(as.data.frame(test_hex[,1]), ensemble_average)
  } else {
    final_submission <- cbind(final_submission, ensemble_average)
  }
  
  write.csv(final_submission, file = "predictionH20_2.csv", quote = F, row.names=F)
  
  print(head(final_submission))
}
cat(paste0("\nOverall cross-validated CMRMSE = " , CMRMSE/length(targets)))

## Write to CSV
write.csv(final_submission, file = "predictionH20_2.csv", quote = F, row.names=F)

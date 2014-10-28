##################
# Use h2o for prediction of soil target values for the Kaggle AfSIS competition. 
# This code is based on the https://github.com/0xdata/h2o/blob/master/R/examples/Kaggle/MeetupKaggleAfricaSoil.R
##################

# Install and load h2o 
install.packages("/Users/carrillo/Programs/h2o-2.7.0.1511/R/h2o_2.7.0.1511.tar.gz",repos = NULL, type = "source")
require( h2o )

# Connect to a remote instance running the same version of h2o 
remoteH2O <- h2o.init(ip = '172.29.13.214', port = 54321, max_mem_size = '60g')

# Load training and test data. 
trainSet <- h2o.importFile(remoteH2O, "/home/carrillo/h2o/data/trainingDecaySubtractedAllPc_matchedWithTestSet.csv")
testSet <- h2o.importFile(remoteH2O, "/home/carrillo/h2o/data/testDecaySubtractedAllPc.csv")

ensemble_size <- 20
n_fold = 20

# Loop through increasing number of principal components used. 
for( pc_count in 2^( seq(1,9,1) ) ) {
  
  # Create working directories 
  dir <- paste( c("/Users/carrillo/workspace/Kaggle/output/AfSIS/h2o/decaySubtractedMatched/",pc_count,"PC/"), collapse="" )
  dir.create( dir )
  setwd( dir )
  
  # Specify predictors (principal components) and target names 
  vars <- colnames(trainSet)
  predictors <- vars[ 2:( pc_count + 1 )]
  targets <- vars[ ( length( vars ) - 4 ) : length( vars ) ]
  
  # Scoring helpers
  MSEs <- matrix(0, nrow = 1, ncol = length(targets))
  RMSEs <- matrix(0, nrow = 1, ncol = length(targets))
  CMRMSE = 0
  
  # Main loop over regression targets
  for ( targetIndex in 1:length( targets ) ) {
    cat("\n\nNow training and cross-validating a DL model for", targets[targetIndex], "...\n")
    
    # Run grid search with n-fold cross-validation
    cvmodel <- h2o.deeplearning( x = predictors, y = targets[targetIndex], data = trainSet,
                                 nfolds = n_fold, classification = F,
                                 activation="RectifierWithDropout",
                                 hidden = c(100,100), hidden_dropout_ratios = c(0.0,0.0), input_dropout_ratio = 0,
                                 epochs = 100, l1 = c(0,1e-5), l2 = c(0,1e-5), rho = 0.99, epsilon = 1e-8, train_samples_per_iteration = -2 )
    
    ## Collect cross-validation error
    MSE <- cvmodel@sumtable[[1]]$prediction_error   #If cvmodel is a grid search model
    RMSE <- sqrt(MSE)
    CMRMSE <- CMRMSE + RMSE #column-mean-RMSE
    MSEs[targetIndex] <- MSE
    RMSEs[targetIndex] <- RMSE
    cat("\nCross-validated MSEs so far:", MSEs)
    cat("\nCross-validated RMSEs so far:", RMSEs)
    cat("\nCross-validated CMRMSE so far:", CMRMSE/targetIndex)
    
    write.csv( RMSEs, file = "errors.csv", quote = F, row.names=F )
    
    
    cat("\n\nTaking parameters from grid search winner for", targets[targetIndex], "...\n")
    p_winning <- cvmodel@sumtable[[ 1 ]]
    
    ## Build an ensemble model on full training data using parameters of the winning model 
    for (n in 1:ensemble_size) {
      cat("\n\nBuilding ensemble model", n, "of", ensemble_size, "for", targets[targetIndex], "...\n")
      model <- h2o.deeplearning(x = predictors, y = targets[targetIndex], key = paste0(targets[targetIndex], "_cv_ensemble_", n, "_of_", ensemble_size),
                                data = trainSet, 
                                classification = F,
                                activation = p_winning$activation, hidden = p_winning$hidden, hidden_dropout_ratios = p_winning$hidden_dropout_ratios,
                                input_dropout_ratio = p_winning$input_dropout_ratio, epochs = p_winning$epochs, l1 = p_winning$l1, l2 = p_winning$l2,
                                rho = p_winning$rho, epsilon = p_winning$epsilon, train_samples_per_iteration = p_winning$train_samples_per_iteration)
      
      ## Aggregate ensemble model predictions
      test_preds <- h2o.predict( model, testSet )
      
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
    
    colnames(ensemble_average)[1] <- targets[targetIndex]
    
    currentSubmission <- cbind(as.data.frame(testSet[,1]), ensemble_average)
    write.csv(currentSubmission, file =paste( "prediction", targets[ targetIndex ], ".csv", sep="", collapse="" ), quote = F, row.names=F)
    
    if (targetIndex == 1) {
      final_submission <- cbind(as.data.frame(testSet[,1]), ensemble_average)
    } else {
      final_submission <- cbind(final_submission, ensemble_average)
    }
    
    write.csv( final_submission, file = "predictionH20.csv", quote = F, row.names=F )
    
    print(head(final_submission))
  }
  cat(paste0("\nOverall cross-validated CMRMSE = " , CMRMSE/length(targets)))
  
  ## Write to CSV
  write.csv(final_submission, file = "predictionH20.csv", quote = F, row.names=F)
} 



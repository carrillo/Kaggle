#######################################
# Best data assuming a base regressor svm.SVR(C=10000.0) with 20-fold CV:
#
# 1. Ca SQRT of MSE: 0.354 (+/- 0.539) (decay subtracted all PC, )
# 2. P SQRT of MSE: 0.779 (+/- 1.115) (decay subtracted 350 PC)
# 3. pH SQRT of MSE: 0.304 (+/- 0.320) (decay subtracted 800 PC)
# 4. SOC SQRT of MSE: 0.221 (+/- 0.259) ( decay subtracted 1250 PC)
# 5. Sand SQRT of MSE: 0.295 (+/- 0.259) (decay subtracted, )
#
#
#######################################

__author__ = 'carrillo'

from sklearn import svm
import pandas as pd
import numpy as np
import ExploreRegressor
from sklearn import linear_model

def reportError( trainFile, testFile, dataId ):
    train = pd.read_csv( trainFile, header=0 )
    test = pd.read_csv( testFile, header=0 )

    # Replace nominal parameters by 0 and 1s
    train.replace( to_replace='Topsoil', value=0, inplace=True )
    train.replace( to_replace='Subsoil', value=1, inplace=True )

    test.replace( to_replace='Topsoil', value=0, inplace=True )
    test.replace( to_replace='Subsoil', value=1, inplace=True )

    est = svm.SVR(C=10000.0, verbose = 0)

    targets = ['Ca','P','pH','SOC','Sand']

    # ests = [
    # svm.SVR(kernel='poly',C=10000.0, degree=2, gamma=0.001 ), # Control SVR for Ca
    # svm.SVR(kernel='rbf',C=15000.0, degree=1, gamma=0.001 ), # Control SVR for P
    # svm.SVR(kernel='rbf',C=10000.0, degree=1, gamma=0.001 ), # Control SVR for pH
    # svm.SVR(kernel='rbf',C=5000.0, degree=1, gamma=0 ), # Control SVR for SOC
    # svm.SVR(kernel='rbf',C=15000.0, degree=1, gamma=0 ), # Control SVR for Sand
    # ]

    # ests = [
    # svm.SVR(kernel='poly',C=9000.0, degree=2, gamma=0.0009 ), # Control SVR for Ca
    # svm.SVR(kernel='rbf',C=17500.0, degree=1, gamma=0.00125 ), # Control SVR for P
    # svm.SVR(kernel='rbf',C=7500.0, degree=1, gamma=0.00075 ), # Control SVR for pH
    # svm.SVR(kernel='rbf',C=5000.0, degree=1, gamma=0 ), # Control SVR for SOC
    # svm.SVR(kernel='rbf',C=12500.0, degree=1, gamma=0 ), # Control SVR for Sand
    # ]

    ests = [
    svm.SVR(C=10000.0), # Control SVR for Ca
    svm.SVR(C=10000.0),  # Control SVR for P
    svm.SVR(C=10000.0),  # Control SVR for pH
    svm.SVR(C=10000.0),  # Control SVR for SOC
    svm.SVR(C=10000.0),  # Control SVR for Sand
    ]

    # ests = [
    # svm.SVR( kernel='poly', C=17000, gamma=0.0075, degree=1 ), # Control SVR for Ca
    # svm.SVR( kernel='rbf', C=11000, gamma=0.0025, degree=1 ), # Control SVR for P
    # svm.SVR( kernel='rbf',C=5750, gamma=0, degree=1 ), # Control SVR for pH
    # svm.SVR( kernel='rbf', C=8250, gamma=0, degree=1 ), # Control SVR for SOC
    # svm.SVR( kernel='rbf', C=20500, gamma=0, degree=1 ), # Control SVR for Sand
    # ]


    errors = []
    for i in range( len(targets) ):
        target = targets[ i ]

        # if( target == 'Ca' ):
        #     continue
        #
        # if( target == 'P' ):
        #     continue
        #
        # if( target == 'pH' ):
        #     continue

        # if( target == 'SOC' ):
        #     continue
        # if( target == 'Sand' ):
        #     continue

        svmReg = ExploreRegressor.ExploreRegressor( train, target, ests[ i ], 0.25 )
        scores = svmReg.reportCrossValidationError( cv=20 )
        errors.append( np.sqrt( -scores.mean() ) )

    print '%s, mean error %0.4f of errors %s' % ( dataId, np.mean( errors ), errors )

# Original data
trainOriginal = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/training.csv'
testOriginal = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/sorted_test.csv'
#reportError(trainOriginal,testOriginal,'original')

# Decay subtracted
trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted_DepthNumeric.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted matched')

# Decay subtracted scaled PC analysis
trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedAllPcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtractedAllPcFromAllFeatures_DepthNumeric.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted all PC matched')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1024PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1024PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 1024 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted512PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted512PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 512 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted256PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted256PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 256 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted128PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted128PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 128 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted64PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted64PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 64 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted32PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted32PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 32 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted16PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted16PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 16 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted8PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted8PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 8 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted4PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted4PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 4 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted2PcFromAllFeatures_matchedWithTestSet.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted2PcFromAllFeatures_DepthNumeric.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 2 PC')


#######################################
# Best data assuming a base regressor svm.SVR(C=10000.0) with 20-fold CV:
#
# 1. Ca SQRT of MSE: 0.354 (+/- 0.539) (decay subtracted all PC, )
# 2. P SQRT of MSE: 0.779 (+/- 1.115) (decay subtracted 350 PC)
# 3. pH SQRT of MSE: 0.304 (+/- 0.320) (decay subtracted 800 PC)
# 4. SOC SQRT of MSE: 0.277 (+/- 0.443) (decay subtracted 1500 PC, )
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

    # ests = [
    # svm.SVR(C=10000.0), # Control SVR for Ca
    # svm.SVR(C=10000.0),  # Control SVR for P
    # svm.SVR(C=10000.0),  # Control SVR for pH
    # svm.SVR(C=10000.0),  # Control SVR for SOC
    # svm.SVR(C=10000.0),  # Control SVR for Sand
    # ]

    ests = [
    svm.SVR( kernel='poly', C=17000, gamma=0.0075, degree=1 ), # Control SVR for Ca
    svm.SVR( kernel='rbf', C=11000, gamma=0.0025, degree=1 ), # Control SVR for P
    svm.SVR( kernel='rbf',C=5750, gamma=0, degree=1 ), # Control SVR for pH
    svm.SVR( kernel='rbf', C=8250, gamma=0, degree=1 ), # Control SVR for SOC
    svm.SVR( kernel='rbf', C=20500, gamma=0, degree=1 ), # Control SVR for Sand
    ]


    errors = []
    for i in range( len(targets) ):
        target = targets[ i ]

        if( target == 'Ca' ):
            continue

        if( target == 'P' ):
            continue

        if( target == 'pH' ):
            continue

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
trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted')

# Decay subtracted scaled PC analysis
trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedAllPc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtractedAllPc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted all PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1500Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1500Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 1500 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1250Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1250Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 1250 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1000Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1000Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 1000 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted875Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted875Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 875 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted800Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted800Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 800 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted750Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted750Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 750 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted700Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted700Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 700 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted625Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted625Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 625 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted500Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted500Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 500 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted400Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted400Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 400 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted350Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted350Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 350 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted300Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted300Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 300 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted250Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted250Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 250 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted200Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted200Pc.csv'
#reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 200 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted100Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted100Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 100 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted50Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted50Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 50 PC')

trainDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted25Pc.csv'
testDecaySubtracted = '/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted25Pc.csv'
reportError(trainDecaySubtracted,testDecaySubtracted,'decay subtracted 25 PC')


#
# # Load data from CSV file
# train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted.csv',header=0)
# test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted.csv',header=0)
#
# train.replace( to_replace='Topsoil', value=0, inplace=True )
# train.replace( to_replace='Subsoil', value=1, inplace=True )
#
# test.replace( to_replace='Topsoil', value=0, inplace=True )
# test.replace( to_replace='Subsoil', value=1, inplace=True )
#
#
# targets = ['Ca','P','pH','SOC','Sand']
#
# errors = []
# for target in targets:
#     svm = ExploreRegressor.ExploreRegressor( train, target, est, 0.25 )
#     scores = svm.reportCrossValidationError( cv=10 )
#     errors.append( -scores.mean() )
#
# print 'Decay substracted, mean error %f of errors %s' % ( np.mean( errors ), errors )

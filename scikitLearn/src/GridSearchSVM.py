from sklearn import svm
from sklearn.grid_search import GridSearchCV

__author__ = 'carrillo'

import pandas as pd
import numpy as np
import ExploreRegressor
from sklearn import linear_model

def improveHyperParameters( train, targetId, tuned_parameter, controlEst ):
    expReg = ExploreRegressor.ExploreRegressor( train, targetId, svm.SVR(), 0.25 )
    best = expReg.gridSearch( tuned_parameters=tuned_parameter, cv=10, verbose=1 )

    print '%s control est' % ( targetId )
    control = ExploreRegressor.ExploreRegressor( train, targetId, controlEst, 0.25 )
    control.reportCrossValidationError( cv=20 )

    print '%s optimal est. Parmeters: %s' % ( targetId, best.best_params_ )
    optimal = ExploreRegressor.ExploreRegressor( train, targetId, best.best_estimator_, 0.25 )
    optimal.reportCrossValidationError( cv=10 )


def getData( fileName ):
    # Load data from CSV file
    data = pd.read_csv( fileName ,header=0)
    data.replace( to_replace='Topsoil', value=0, inplace=True )
    data.replace( to_replace='Subsoil', value=1, inplace=True )
    return data


# Load training data from CSV files
ca_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedAllPc.csv')
p_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted350Pc.csv')
pH_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted800Pc.csv')
soc_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1500Pc.csv')
sand_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted.csv')

data = [ ca_train, p_train, pH_train, soc_train, sand_train ]

targets = ['Ca','P','pH','SOC','Sand']


tuned_parameterList = [
    [{'kernel': ['poly'], 'C': [17000] , 'degree': [1], 'gamma': [0.0075] } ], # Parameter tuning for Ca
    [{'kernel': ['rbf'], 'C': [11000] , 'degree': [1], 'gamma': [0.0025] } ], # Parameter tuning for P
    [{'kernel': ['rbf'], 'C': [5750] , 'degree': [1], 'gamma': [0] } ], # Parameter tuning for pH
    [{'kernel': ['rbf'], 'C': [8250] , 'degree': [1], 'gamma': [0] } ], # Parameter tuning for SOC
    [{'kernel': ['rbf'], 'C': [20500] , 'degree': [1], 'gamma': [0] } ] # Parameter tuning for Sand
]


# Regular est from 20 fold cv
# controlEsts = [
#     svm.SVR(kernel='poly',C=5000.0, degree=2, gamma=0.0009 ), # Control SVR for Ca
#     svm.SVR(kernel='rbf',C=20000.0, degree=1, gamma=0.00125 ), # Control SVR for P
#     svm.SVR(kernel='rbf',C=6100.0, degree=1, gamma=0.000875 ), # Control SVR for pH
#     svm.SVR(kernel='rbf',C=5000.0, degree=1, gamma=0 ), # Control SVR for SOC
#     svm.SVR(kernel='rbf',C=10500.0, degree=1, gamma=0 ), # Control SVR for Sand
# ]

# Regular est from 20 fold cv
controlEsts = [
    svm.SVR( kernel='poly', C=17000, gamma=0.0075, degree=1 ), # Control SVR for Ca
    svm.SVR( kernel='rbf', C=11000, gamma=0.0025, degree=1 ), # Control SVR for P
    svm.SVR( kernel='rbf',C=5750, gamma=0, degree=1 ), # Control SVR for pH
    svm.SVR( kernel='rbf', C=8250, gamma=0, degree=1 ), # Control SVR for SOC
    svm.SVR( kernel='rbf', C=20500, gamma=0, degree=1 ), # Control SVR for Sand
]



for i in range( len(targets) ):
    target = targets[ i ]
    tuned_parameters = tuned_parameterList[ i ]
    controlEst = controlEsts[ i ]
    improveHyperParameters( data[ i ], target, tuned_parameters, controlEst )
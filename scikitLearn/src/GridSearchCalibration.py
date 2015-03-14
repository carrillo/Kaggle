from sklearn import svm
from sklearn.grid_search import GridSearchCV

__author__ = 'carrillo'

import pandas as pd
import numpy as np
import ExploreRegressor
from sklearn import linear_model

def improveHyperParameters( train, targetId, tuned_parameter, controlEst ):
    # Shuffle rows to not over fit on groups
    train = train.reindex( np.random.permutation(train.index) )

    expReg = ExploreRegressor.ExploreRegressor( train, targetId, linear_model.Ridge(), 0.25 )
    best = expReg.gridSearch( tuned_parameters=tuned_parameter, cv=50, verbose=1 )

    print '%s control est' % ( targetId )
    control = ExploreRegressor.ExploreRegressor( train, targetId, controlEst, 0.25 )
    control.reportCrossValidationError( cv=10 )

    print '%s optimal est. Parmeters: %s' % ( targetId, best.best_params_ )
    optimal = ExploreRegressor.ExploreRegressor( train, targetId, best.best_estimator_, 0.25 )
    optimal.reportCrossValidationError( cv=10 )



# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM9_calibration_addedTarget.csv',header=0)

targets = ['Ca','P','pH','SOC','Sand']

tuned_parameterList = [
    [{'alpha': [0.00001,0.0001,0.001] }], # Parameter tuning for Ca
    [{'alpha': [0.5,1,5] }], # Parameter tuning for P
    [{'alpha': [0.5,1,5] }], # Parameter tuning for pH
    [{'alpha': [0.00001,0.0001,0.001] }],  # Parameter tuning for SOC
    [{'alpha': [0.05,0.1,0.5] }], # Parameter tuning for Sand
]


# Regular est from 20 fold cv
controlEsts = [
    linear_model.Ridge( alpha=0.001 ), # Control SVR for Ca
    linear_model.Ridge(),  # Control SVR for P
    linear_model.Ridge(),  # Control SVR for pH
    linear_model.Ridge(),  # Control SVR for SOC
    linear_model.Ridge(),  # Control SVR for Sand
]


for i in range( len(targets) ):
    target = targets[ i ]


    tuned_parameters = tuned_parameterList[ i ]
    controlEst = controlEsts[ i ]
    improveHyperParameters( train, target, tuned_parameters, controlEst )
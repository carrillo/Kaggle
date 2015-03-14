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
    best = expReg.gridSearch( tuned_parameters=tuned_parameter, cv=50, verbose=2 )

    print '%s control est' % ( targetId )
    control = ExploreRegressor.ExploreRegressor( train, targetId, controlEst, 0.25 )
    control.reportCrossValidationError( cv=10 )

    print '%s optimal est. Parmeters: %s' % ( targetId, best.best_params_ )
    optimal = ExploreRegressor.ExploreRegressor( train, targetId, best.best_estimator_, 0.25 )
    optimal.reportCrossValidationError( cv=10 )



# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedSmooth5e-04.csv',header=0)

train.replace( to_replace='Topsoil', value=0, inplace=True )
train.replace( to_replace='Subsoil', value=1, inplace=True )


targets = ['Ca','P','pH','SOC','Sand']

tuned_parameterList = [
    [{'alpha': [5e-02,6e-02,7e-02,8e-02,9e-02,1e-01,1.1e-01,1.2e-01,1.3e-01,1.4e-01,1.5e-01] } ], # Parameter tuning for Ca
    [{'alpha': [5e-02,6e-02,7e-02,8e-02,9e-02,1e-01,1.1e-01,1.2e-01,1.3e-01,1.4e-01,1.5e-01] } ], # Parameter tuning for P
    [{'alpha': [5e-04,6e-04,7e-04,8e-04,9e-04,1e-03,1.1e-03,1.2e-03,1.3e-03,1.4e-03,1.5e-03] } ], # Parameter tuning for pH
    [{'alpha': [5e-02,6e-02,7e-02,8e-02,9e-02,1e-01,1.1e-01,1.2e-01,1.3e-01,1.4e-01,1.5e-01] } ], # Parameter tuning for SOC
    [{'alpha': [5e-03,6e-03,7e-03,8e-03,9e-03,1e-02,1.1e-02,1.2e-02,1.3e-02,1.4e-02,1.5e-02] } ], # Parameter tuning for Sand
]


# Regular est from 20 fold cv
controlEsts = [
    linear_model.Ridge( alpha=0.06 ), # Control SVR for Ca
    linear_model.Ridge( alpha=0.05 ), # Control SVR for P
    linear_model.Ridge( alpha=0.0015 ), # Control SVR for pH
    linear_model.Ridge(  alpha=0.05 ), # Control SVR for SOC
    linear_model.Ridge( alpha=0.015 ), # Control SVR for Sand
]

targetId = 'SOC'
print '%s control est' % ( targetId )
control = ExploreRegressor.ExploreRegressor( train, targetId, linear_model.Ridge(   alpha=0.1  ), 0.25 )
control.reportCrossValidationError( cv=100 )

print '%s optimal est. Parmeters: %s' % ( targetId, 'optimal' )
optimal = ExploreRegressor.ExploreRegressor( train, targetId, linear_model.Ridge( alpha=0.05 ), 0.25 )
optimal.reportCrossValidationError( cv=100 )


for i in range( len(targets) ):
    target = targets[ i ]

    if( target == 'Ca' ):
        continue

    if( target == 'P' ):
        continue

    if( target == 'pH' ):
        continue

    if( target == 'SOC' ):
        continue


    tuned_parameters = tuned_parameterList[ i ]
    controlEst = controlEsts[ i ]
    improveHyperParameters( train, target, tuned_parameters, controlEst )
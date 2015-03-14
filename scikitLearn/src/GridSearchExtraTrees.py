from sklearn import svm
from sklearn.grid_search import GridSearchCV

__author__ = 'carrillo'

import pandas as pd
import numpy as np
import ExploreRegressor
from sklearn import ensemble

def improveHyperParameters( train, targetId, tuned_parameter, controlEst ):
    # Shuffle rows to not over fit on groups
    train = train.reindex( np.random.permutation(train.index) )

    expReg = ExploreRegressor.ExploreRegressor( train, targetId, ensemble.ExtraTreesRegressor( n_jobs=10 ), 0.25 )
    best = expReg.gridSearch( tuned_parameters=tuned_parameter, cv=5, verbose=2 )

    print '%s control est' % ( targetId )
    control = ExploreRegressor.ExploreRegressor( train, targetId, controlEst, 0.25 )
    control.reportCrossValidationError( cv=10 )

    print '%s optimal est. Parmeters: %s' % ( targetId, best.best_params_ )
    optimal = ExploreRegressor.ExploreRegressor( train, targetId, best.best_estimator_, 0.25 )
    optimal.reportCrossValidationError( cv=10 )



# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted.csv',header=0)

train.replace( to_replace='Topsoil', value=0, inplace=True )
train.replace( to_replace='Subsoil', value=1, inplace=True )


targets = ['Ca','P','pH','SOC','Sand']

tuned_parameterList = [
    [{'n_estimators': [250,500,750], 'max_depth': [4,8], 'n_jobs' : [1], 'min_samples_leaf' : [2,4,8,16]  } ], # Parameter tuning for Ca
    [{'n_estimators': [25,50,75], 'max_depth': [None,8], 'n_jobs' : [1], 'min_samples_leaf' : [3,4,5]  } ], # Parameter tuning for P
    [{'n_estimators': [250,500,750], 'max_depth': [None,2,4,8], 'n_jobs' : [1], 'min_samples_leaf' : [1,2,3]  } ], # Parameter tuning for pH
    [{'n_estimators': [250,500,750], 'max_depth': [None,2,4,8], 'n_jobs' : [1], 'min_samples_leaf' : [1,2,3]  } ], # Parameter tuning for SOC
    [{'n_estimators': [250,500,750], 'max_depth': [None,2,4,8], 'n_jobs' : [1], 'min_samples_leaf' : [1,2,3]  } ], # Parameter tuning for Sand
]


# Regular est from 20 fold cv
controlEsts = [
    ensemble.ExtraTreesRegressor( n_jobs=1, n_estimators=500, max_depth=8, min_samples_leaf=8 ), # Control SVR for Ca
    ensemble.ExtraTreesRegressor( n_jobs=1, n_estimators=50, max_depth=None, min_samples_leaf=4 ), # Control SVR for P
    ensemble.ExtraTreesRegressor( n_jobs=1, n_estimators=500, max_depth=None, min_samples_leaf=2 ), # Control SVR for pH
    ensemble.ExtraTreesRegressor( n_jobs=1, n_estimators=500, max_depth=None, min_samples_leaf=2 ), # Control SVR for SOC
    ensemble.ExtraTreesRegressor( n_jobs=1, n_estimators=500, max_depth=None, min_samples_leaf=2 ) # Control SVR for Sand
]
#
# targetId = 'SOC'
# print '%s control est' % ( targetId )
# control = ExploreRegressor.ExploreRegressor( train, targetId, linear_model.Ridge(   alpha=0.1  ), 0.25 )
# control.reportCrossValidationError( cv=100 )
#
# print '%s optimal est. Parmeters: %s' % ( targetId, 'optimal' )
# optimal = ExploreRegressor.ExploreRegressor( train, targetId, linear_model.Ridge( alpha=0.05 ), 0.25 )
# optimal.reportCrossValidationError( cv=100 )


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
    #
    # if( target == 'SOC' ):
    #     continue


    tuned_parameters = tuned_parameterList[ i ]
    controlEst = controlEsts[ i ]
    improveHyperParameters( train, target, tuned_parameters, controlEst )
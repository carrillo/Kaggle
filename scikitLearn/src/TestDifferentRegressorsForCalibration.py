from sklearn.linear_model import OrthogonalMatchingPursuit

__author__ = 'carrillo'

########################################
#
# Ca:
#   1. NearestNeighbor 2 model Ca MSE: 0.126 (+/- 0.063) <- optimal already, use SVM with rbf instead
#   2. RidgeRegression model Ca MSE: 0.139 (+/- 0.067)
#   3. RandomForestRegressor Ca MSE: 0.147 (+/- 0.093)
# P:
#   1. NearestNeighbor 2 model: P MSE: 0.811 (+/- 0.297) <- optimal already, use SVM with rbf instead
#   2. ExtraTreesRegressor: P MSE: 0.839 (+/- 0.277)
#   3. RidgeRegression model P MSE: 0.873 (+/- 0.290)
# pH:
#   1. RidgeRegressionCV model pH MSE: 0.120 (+/- 0.068)
#   2. BayesianRidgeRegression model pH MSE: 0.136 (+/- 0.071)
#   3. ExtraTreesRegressor model: pH MSE: 0.223 (+/- 0.093)
# SOC
#   1. BayesianRidgeRegression model SOC MSE: 0.113 (+/- 0.110)
#   2. RidgeRegressionCV model SOC MSE: 0.116 (+/- 0.107)
#   3. ExtraTreesRegressor model SOC MSE: 0.155 (+/- 0.080)
# Sand
#   1. BayesianRidgeRegression model Sand MSE: 0.120 (+/- 0.044)
#   2. RidgeRegressionCV model Sand MSE: 0.120 (+/- 0.040)
#   3. ExtraTreesRegressor model: Sand MSE: 0.173 (+/- 0.071)
########################################



import pandas as pd
import ExploreRegressor
from sklearn import linear_model
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import tree
from sklearn import ensemble

def testRegressor( train, regressor, target, id ):
    expReg = ExploreRegressor.ExploreRegressor( train, target, regressor, 0.25 )
    print '%s model' % ( id )
    expReg.reportCrossValidationError( cv=20 )

# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM9_calibration_addedTarget.csv',header=0)

# train.replace( to_replace='Topsoil', value=0, inplace=True )
# train.replace( to_replace='Subsoil', value=1, inplace=True )

target = 'P'

# Linear models
testRegressor( train, linear_model.LinearRegression(), target, 'LinearRegression' )
testRegressor( train, linear_model.Ridge(), target, 'RidgeRegression' )
testRegressor( train, linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 2, 4, 8, 16, 32]), target, 'RidgeRegressionCV' )

testRegressor( train, linear_model.Lasso(), target, 'Lasso' )
testRegressor( train, linear_model.LassoLars(), target, 'LassoLars' )

testRegressor( train, OrthogonalMatchingPursuit() , target, 'OMP' )

# Stochastic gradient descent
testRegressor( train, linear_model.SGDRegressor( loss='squared_loss' ), target, 'SGDRegressor squared loss' )

# Bayesian approaches
testRegressor( train, linear_model.BayesianRidge() , target, 'BayesianRidgeRegression' )
#testRegressor( train, ARDRegression() , target, 'ARDRegression' )



testRegressor( train, linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive') , target, 'PassiveAggressiveRegressor' )
testRegressor( train, linear_model.PassiveAggressiveRegressor(loss='squared_epsilon_insensitive') , target, 'PassiveAggressiveRegressor squared loss' )


# Support Vector machines
testRegressor( train, svm.SVR(kernel='poly'), target, 'SVM poly' )
testRegressor( train, svm.SVR(kernel='rbf'), target, 'SVM rbf' )
testRegressor( train, svm.SVR(kernel='sigmoid'), target, 'SVM sigmoid' )

# Nearest neighbors
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=1 ), target, 'NearestNeighbor 1' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=2 ), target, 'NearestNeighbor 2' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=3 ), target, 'NearestNeighbor 3' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=4 ), target, 'NearestNeighbor 4' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=8 ), target, 'NearestNeighbor 8' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=16 ), target, 'NearestNeighbor 16' )
testRegressor( train, neighbors.KNeighborsRegressor( n_neighbors=32 ), target, 'NearestNeighbor 32' )


# Gaussian process
# testRegressor( train, gaussian_process.GaussianProcess(), target, 'Gaussian process' )

# Regression trees
testRegressor( train, tree.DecisionTreeRegressor(), target, 'Regression tree' )
testRegressor( train, ensemble.RandomForestRegressor(n_estimators=1000), target, 'RandomForestRegressor' )
testRegressor( train, ensemble.ExtraTreesRegressor(), target, 'ExtraTreesRegressor' )

# Gradient tree Boosting
#testRegressor( train, ensemble.GradientBoostingRegressor(loss='ls'), target, 'Gradient tree boosting' )
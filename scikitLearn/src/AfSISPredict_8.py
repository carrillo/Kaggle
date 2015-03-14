from sklearn import svm

__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Prediction
import datetime
import time
import matplotlib.pyplot as plt

################
# For first submission use all features with learned best estimator.
# Use
################


# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtractedSmooth.csv',header=0)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtractedSmooth.csv',header=0)

train.replace( to_replace='Topsoil', value=0, inplace=True )
train.replace( to_replace='Subsoil', value=1, inplace=True )

test.replace( to_replace='Topsoil', value=0, inplace=True )
test.replace( to_replace='Subsoil', value=1, inplace=True )


# Set up the individual regressors
ca_est = svm.SVR(kernel='poly',C=10000.0, degree=2, gamma=0.001 )
ca_featureSelection = None
ca_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'Ca' )

ca_predict.learn()
ca_prediction = ca_predict.predict()


# Predict P
p_est = svm.SVR(kernel='rbf',C=15000.0, degree=1, gamma=0.001 )
p_featureSelection = None
p_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_est = svm.SVR(kernel='rbf',C=10000.0, degree=1, gamma=0.001 )
pH_featureSelection = None
pH_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_est = svm.SVR(kernel='rbf',C=5000.0, degree=1, gamma=0 )
soc_featureSelection = None
soc_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_est = svm.SVR(kernel='rbf',C=15000.0, degree=1, gamma=0 )
sand_featureSelection = None
sand_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'Sand' )
sand_predict.learn()
sand_prediction = sand_predict.predict()


# Merge data into one data frame and order
predict = pd.merge( ca_prediction, p_prediction, on='PIDN' )
predict = pd.merge( predict, pH_prediction, on='PIDN' )
predict = pd.merge( predict, soc_prediction, on='PIDN' )
predict = pd.merge( predict, sand_prediction, on='PIDN' )
predict = predict.reindex_axis( ['PIDN','Ca','P','pH','SOC','Sand'], axis=1 )


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM8_' + st + ".csv"

predict.to_csv(fileOut,header=True,index=False)

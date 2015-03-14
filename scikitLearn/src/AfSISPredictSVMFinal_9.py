from sklearn import svm

__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Prediction
import datetime
import time
import matplotlib.pyplot as plt


# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted.csv',header=0)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted.csv',header=0)

train.replace( to_replace='Topsoil', value=0, inplace=True )
train.replace( to_replace='Subsoil', value=1, inplace=True )
trainForCalibration = train.drop( ['Ca','P','pH','SOC','Sand'], 1 )


test.replace( to_replace='Topsoil', value=0, inplace=True )
test.replace( to_replace='Subsoil', value=1, inplace=True )


# Set up the individual regressors
ca_est = svm.SVR( kernel='poly',C=5000.0, degree=2, gamma=0.0009 ) # Control SVR for Ca
ca_featureSelection = None
ca_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'Ca' )

ca_predict.learn()
ca_prediction = ca_predict.predict()


# Predict P
p_est = svm.SVR(kernel='rbf',C=20000.0, degree=1, gamma=0.00125 ) # Control SVR for P
p_featureSelection = None
p_predict = Prediction.Prediction( train, test, p_est, p_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_est = svm.SVR(kernel='rbf',C=6100.0, degree=1, gamma=0.000875 ) # Control SVR for pH
pH_featureSelection = None
pH_predict = Prediction.Prediction( train, test, pH_est, pH_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_est = svm.SVR(kernel='rbf',C=5000.0, degree=1, gamma=0 ) # Control SVR for SOC
soc_featureSelection = None
soc_predict = Prediction.Prediction( train, test, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_est = svm.SVR(kernel='rbf',C=10500.0, degree=1, gamma=0 ) # Control SVR for Sand
sand_featureSelection = None
sand_predict = Prediction.Prediction( train, test, sand_est, sand_featureSelection, 'Sand' )
sand_predict.learn()
sand_prediction = sand_predict.predict()


# Merge data into one data frame and order
predict = pd.merge( ca_prediction, p_prediction, on='PIDN' )
predict = pd.merge( predict, pH_prediction, on='PIDN' )
predict = pd.merge( predict, soc_prediction, on='PIDN' )
predict = pd.merge( predict, sand_prediction, on='PIDN' )
predict = predict.reindex_axis( ['PIDN','Ca','P','pH','SOC','Sand'], axis=1 )


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM9_' + st + ".csv"

predict.to_csv(fileOut,header=True,index=False)

# Write prediction for calibration
ca_predict = Prediction.Prediction( train, trainForCalibration, ca_est, ca_featureSelection, 'Ca' )
ca_predict.learn()
ca_calibration = ca_predict.predict()

p_predict = Prediction.Prediction( train, trainForCalibration, p_est, p_featureSelection, 'P' )
p_predict.learn()
p_calibration = p_predict.predict()

pH_predict = Prediction.Prediction( train, trainForCalibration, pH_est, pH_featureSelection, 'pH' )
pH_predict.learn()
pH_calibration = pH_predict.predict()

soc_predict = Prediction.Prediction( train, trainForCalibration, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_calibration = soc_predict.predict()

sand_predict = Prediction.Prediction( train, trainForCalibration, sand_est, sand_featureSelection, 'Sand' )
sand_predict.learn()
sand_calibration = sand_predict.predict()

calibration = pd.merge( ca_calibration, p_calibration, on='PIDN' )
calibration = pd.merge( calibration, pH_calibration, on='PIDN' )
calibration = pd.merge( calibration, soc_calibration, on='PIDN' )
calibration = pd.merge( calibration, sand_calibration, on='PIDN' )
calibration = calibration.reindex_axis( ['PIDN','Ca','P','pH','SOC','Sand'], axis=1 )

fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM9_' + "calibration" + ".csv"
calibration.to_csv(fileOut,header=True,index=False)
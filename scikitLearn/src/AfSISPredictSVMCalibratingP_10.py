from sklearn import svm

__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Prediction
import datetime
import time
from sklearn import linear_model
import matplotlib.pyplot as plt


# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM10_calibration_addedTarget.csv',header=0)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM10_2014-10-12-19_01_38.csv',header=0)


# Set up the individual regressors
ca_est = linear_model.Ridge() # Control SVR for Ca
ca_featureSelection = None
ca_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'Ca' )

ca_predict.learn()
ca_prediction = ca_predict.predict()


# Predict P
p_est = linear_model.Ridge() # Control SVR for P
p_featureSelection = None
p_predict = Prediction.Prediction( train, test, p_est, p_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_est = linear_model.Ridge() # Control SVR for pH
pH_featureSelection = None
pH_predict = Prediction.Prediction( train, test, pH_est, pH_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_est = linear_model.Ridge() # Control SVR for SOC
soc_featureSelection = None
soc_predict = Prediction.Prediction( train, test, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_est = linear_model.Ridge() # Control SVR for Sand
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
fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM10WithCalibration_' + st + ".csv"

predict.to_csv(fileOut,header=True,index=False)

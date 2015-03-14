__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Prediction
import datetime
import time
import matplotlib.pyplot as plt

################
# For first submission use all features with learned best estimator.
################


# Load data from CSV file
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingTransformed.csv',header=0)
train = train.drop('Ptransformed',1)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testTransformed.csv',header=0)

# Set up the individual regressors
ca_est = RandomForestRegressor( n_estimators=800, max_depth=7, min_samples_leaf=4, n_jobs=10, oob_score=True, verbose=0 )
ca_featureSelection = ['id', 'X3.1065_PeakPosY', 'LSTN', 'X3.718_PeakPosY', 'ELEV', 'X3.2235_PeakPosY', 'quantile80', 'X3.0995_PeakPosY', 'X3.0105_PeakPosY', 'X3.3795_PeakPosY', 'X3.077_PeakPosY', 'X3.4015_PeakPosY', 'X3.6595_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
ca_predict = Prediction.Prediction( train, test, ca_est, ca_featureSelection, 'Ca' )

ca_predict.learn()
ca_prediction = ca_predict.predict()

# # Test if the prediction is for the right id
# testFit = pd.merge( train.ix()[:,['id','Ca'] ], ca_prediction, left_on='id', right_on='PIDN' )
# ca_obs = testFit.ix()[:,'Ca_x']
# ca_pred = testFit.ix()[:,'Ca_y']
# plt.figure()
# plt.scatter( ca_obs, ca_pred, c="k", label="data")
# plt.xlabel("observed")
# plt.ylabel("predicted")
# plt.legend()
# plt.show()

# Predict P
p_est = RandomForestRegressor( n_estimators=800, max_depth=None, min_samples_leaf=2, n_jobs=10, oob_score=True, verbose=0 )
p_featureSelection = ['id', 'LSTN', 'X3.0995_PeakPosY', 'X3.077_PeakPosY', 'X3.3695_PeakPosY', 'X3.298_PeakPosY', 'X3.2125_PeakPosY', 'X3_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
p_predict = Prediction.Prediction( train, test, p_est, p_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
pH_featureSelection = ['id', 'EVI', 'X3.1065_PeakPosY', 'REF7', 'X3.718_PeakPosY', 'X3.4015_PeakPosY', 'X3.2235_PeakPosY', 'TMAP', 'BSAV', 'LSTD', 'Ca', 'P', 'pH', 'SOC', 'Sand']
pH_predict = Prediction.Prediction( train, test, pH_est, pH_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_est = RandomForestRegressor( n_estimators=400, max_depth=None, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
soc_featureSelection = ['id', 'X2.849_PeakPosY', 'X3.2715_PeakPosY', 'REF7', 'X3.6595_PeakPosY', 'X3.6005_PeakPosY', 'X3.4015_PeakPosY', 'X2.877_PeakPosY', 'X2.8225_PeakPosY', 'X3.1825_PeakPosY', 'X3.6085_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
soc_predict = Prediction.Prediction( train, test, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_est = RandomForestRegressor( n_estimators=600, max_depth=4, min_samples_leaf=8, n_jobs=10, oob_score=True, verbose=0 )
sand_featureSelection = ['id', 'X3.077_PeakPosY', 'X3.1065_PeakPosY', 'X3.2715_PeakPosY', 'X3.298_PeakPosY', 'TMFI', 'TMAP', 'X3.0635_PeakPosY', 'X3.2555_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
sand_predict = Prediction.Prediction( train, test, sand_est, sand_featureSelection, 'Sand' )
sand_predict.learn()
sand_prediction = sand_predict.predict()


# Merge data into one data frame and order
predict = pd.merge( ca_prediction, p_prediction, on='PIDN' )
predict = pd.merge( predict, pH_prediction, on='PIDN' )
predict = pd.merge( predict, soc_prediction, on='PIDN' )
predict = pd.merge( predict, sand_prediction, on='PIDN' )
predict = predict.reindex_axis(['PIDN','Ca','P','pH','SOC','Sand'], axis=1)


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictRegressionTree2_' + st + ".csv"

predict.to_csv(fileOut,header=True,index=False)

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
train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingRenamedId.csv',header=0)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/sorted_testRenamedId.csv',header=0)

train.replace( to_replace='Topsoil', value=0, inplace=True )
train.replace( to_replace='Subsoil', value=1, inplace=True )

test.replace( to_replace='Topsoil', value=0, inplace=True )
test.replace( to_replace='Subsoil', value=1, inplace=True )



# Set up the individual regressors
ca_est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True )
ca_featureSelection = None
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
p_est = RandomForestRegressor( n_estimators=400, max_depth=4, min_samples_leaf=8, n_jobs=10, oob_score=True )
p_featureSelection = None
p_predict = Prediction.Prediction( train, test, p_est, p_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True )
pH_featureSelection = None
pH_predict = Prediction.Prediction( train, test, pH_est, pH_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True )
soc_featureSelection = None
soc_predict = Prediction.Prediction( train, test, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True )
sand_featureSelection = None
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
fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictRegressionTree3_' + st + ".csv"

predict.to_csv(fileOut,header=True,index=False)

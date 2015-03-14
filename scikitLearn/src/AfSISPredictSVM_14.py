from sklearn import svm

#############################
#
# Use training set matched to testing set (nearest neighbor)
#
# CV (20-fold) predicts: 0.342
#
# Ca SQRT of MSE: 0.217 (+/- 0.269) (decay subtracted 1024 PC)
# P SQRT of MSE: 0.663 (+/- 1.007) (decay subtracted 512 PC)
# pH SQRT of MSE: 0.299 (+/- 0.254) (decay subtracted 1024 PC)
# SOC SQRT of MSE: 0.221 (+/- 0.259) (decauy subtracted 1250PC)
# Sand SQRT of MSE: 0.317 (+/- 0.307) (decay subtracted matched)
#
############################

__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Prediction
import datetime
import time
import matplotlib.pyplot as plt

def getData( fileName ):
    data = pd.read_csv( fileName ,header=0)
    data.replace( to_replace='Topsoil', value=0, inplace=True )
    data.replace( to_replace='Subsoil', value=1, inplace=True )
    return data

# Set up the individual regressors
ca_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1024PcFromAllFeatures_matchedWithTestSet.csv')
ca_test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1024PcFromAllFeatures_DepthNumeric.csv')
ca_est = svm.SVR(C=10000.0) # Control SVR for Ca
ca_featureSelection = None
ca_predict = Prediction.Prediction( ca_train, ca_test, ca_est, ca_featureSelection, 'Ca' )

ca_predict.learn()
ca_prediction = ca_predict.predict()


# Predict P
p_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted512PcFromAllFeatures_matchedWithTestSet.csv')
p_test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted512PcFromAllFeatures_DepthNumeric.csv')
p_est = svm.SVR(C=10000.0) # Control SVR for P
p_featureSelection = None
p_predict = Prediction.Prediction( p_train, p_test, p_est, p_featureSelection, 'P' )

p_predict.learn()
p_prediction = p_predict.predict()

# Predict pH
pH_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1024PcFromAllFeatures_matchedWithTestSet.csv')
pH_test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1024PcFromAllFeatures_DepthNumeric.csv')
pH_est = svm.SVR(C=10000.0) # Control SVR for pH
pH_featureSelection = None
pH_predict = Prediction.Prediction( pH_train, pH_test, pH_est, pH_featureSelection, 'pH' )

pH_predict.learn()
pH_prediction = pH_predict.predict()

# Predict SOC
soc_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1024PcFromAllFeatures_matchedWithTestSet.csv')
soc_test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1024PcFromAllFeatures_DepthNumeric.csv')
soc_est = svm.SVR(C=10000.0) # Control SVR for SOC
soc_featureSelection = None
soc_predict = Prediction.Prediction( soc_train, soc_test, soc_est, soc_featureSelection, 'SOC' )
soc_predict.learn()
soc_prediction = soc_predict.predict()

# Predict Sand
sand_train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingDecaySubtracted1024PcFromAllFeatures_matchedWithTestSet.csv')
sand_test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testDecaySubtracted1024PcFromAllFeatures_DepthNumeric.csv')
sand_est = svm.SVR(C=10000.0) # Control SVR for Sand
sand_featureSelection = None
sand_predict = Prediction.Prediction( sand_train, sand_test, sand_est, sand_featureSelection, 'Sand' )
sand_predict.learn()
sand_prediction = sand_predict.predict()


# Merge data into one data frame and order
predict = pd.merge( ca_prediction, p_prediction, on='PIDN' )
predict = pd.merge( predict, pH_prediction, on='PIDN' )
predict = pd.merge( predict, soc_prediction, on='PIDN' )
predict = pd.merge( predict, sand_prediction, on='PIDN' )
predict = predict.reindex_axis( ['PIDN','Ca','P','pH','SOC','Sand'], axis=1 )


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')

fileOut = '/Users/carrillo/workspace/Kaggle/output/AfSIS/predictSVM14_' + st + ".csv"
predict.to_csv(fileOut,header=True,index=False)

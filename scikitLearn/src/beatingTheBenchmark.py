# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation

train = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingTransformedPIDN.csv',header=0)
train = train.drop('Ptransformed',1)
test = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/testTransformedPIDN.csv',header=0)


labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]


sup_vec = svm.SVR(C=10000.0, verbose = 2)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

sample = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('/Users/carrillo/workspace/Kaggle/output/AfSIS/beating_benchmarkModified.csv', index = False)


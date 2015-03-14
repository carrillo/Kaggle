__author__ = 'carrillo'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import ExploreRegressor

def exploreResult( reg, plot ):
    # Explore result
    reg.reportFeatureImportance( plot=False )

    reg.reportTrainingPerformance()
    reg.reportTestPerformance()

    if( plot ):
        reg.plotTrainingError()
        reg.plotTestError()

# Load data from CSV file
df = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingTransformed.csv',header=0)

# Specify SOC regressor
targetId = 'P'
est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regAll = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regAll.getErrors( iterations=20 )
print 'All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

clv = regAll.gridSearch( verbose = 0 )
regNew= ExploreRegressor.ExploreRegressor( df, targetId, clv.best_estimator_, 0.25 )
regNew.learn()
trainMSE, testMSE = regNew.getErrors( iterations=20 )
print '%f\t%f\t%f\t%f\t%s' % ( np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ),
clv.best_params_ )

featureSelection = ['id', 'LSTN', 'X3.0995_PeakPosY', 'X3.077_PeakPosY', 'X3.3695_PeakPosY', 'X3.298_PeakPosY', 'X3.2125_PeakPosY', 'X3_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
reducedPEst = RandomForestRegressor( n_estimators=800, max_depth=None, min_samples_leaf=2, n_jobs=10, oob_score=True, verbose=0 )
regPReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, reducedPEst, 0.25 )
#clv = regPReduced.gridSearch( verbose= 2 )
trainMSE, testMSE = regPReduced.getErrors( iterations=20 )
print 'P best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f\t%s' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), clv.best_params_ )




print 'featureCount\tmeanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE\tBestParameters\tSelectedFeatures'

for featureCount in range(5,15):
    featureSelection = regAll.getImportantFeatures( featuresToRetrieve=featureCount, learningIterations=10, featuresToCollectPerIteration=( featureCount*2 ), verbose=0 )
    #featureSelection = ['id', 'X3.077_PeakPosY', 'X3.298_PeakPosY', 'LSTN', 'X3.0995_PeakPosY', 'X3.3695_PeakPosY', 'quantile40', 'quantile20', 'X3.718_PeakPosY', 'CTI', 'BSAN', 'Ca', 'P', 'pH', 'SOC', 'Sand']
    #featureSelection = ['id', 'X3.077_PeakPosY', 'LSTN', 'X3.298_PeakPosY', 'X3.0995_PeakPosY', 'X3.3695_PeakPosY', 'X3.718_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']


    dfNew = df.ix()[:,featureSelection]
    regPNew = ExploreRegressor.ExploreRegressor( dfNew, targetId, est, 0.25 )
    #clv = regPNew.gridSearch( verbose = 0 )
    bestEst = RandomForestRegressor( verbose=0, n_jobs=10, oob_score=True,
                                 n_estimators=800, min_samples_leaf=2, max_depth=None )

    regPNew = ExploreRegressor.ExploreRegressor( dfNew, targetId, bestEst, 0.25 )
    regPNew.learn()

    trainMSE, testMSE = regPNew.getErrors( iterations=20 )

    print '%d\t%f\t%f\t%f\t%f\t%s' % ( featureCount, np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), featureSelection )
    #print '%d\t%f\t%f\t%f\t%f\t%s\t%s' % ( featureCount, np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), clv.best_params_, featureSelection )


# Specify P regressor
targetId = 'P'
est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regPAll = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regPAll.getErrors( iterations=20 )
print 'All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

featureSelection = ['id', 'LSTN', 'X3.0995_PeakPosY', 'X3.077_PeakPosY', 'X3.3695_PeakPosY', 'X3.298_PeakPosY', 'X3.2125_PeakPosY', 'X3_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
reducedPEst = RandomForestRegressor( n_estimators=800, max_depth=None, min_samples_leaf=2, n_jobs=10, oob_score=True, verbose=0 )
regPReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, reducedPEst, 0.25 )
#clv = regPReduced.gridSearch( verbose= 2 )
trainMSE, testMSE = regPReduced.getErrors( iterations=20 )
print 'P best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), )

# Specify Ca regressor. The next best estimator uses smaller feature set, but results in a slightly worse score.
targetId = 'Ca'
est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regAll = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regAll.getErrors( iterations=20 )
print 'Ca All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

featureSelection = ['id', 'X3.1065_PeakPosY', 'LSTN', 'X3.718_PeakPosY', 'ELEV', 'X3.2235_PeakPosY', 'quantile80', 'X3.0995_PeakPosY', 'X3.0105_PeakPosY', 'X3.3795_PeakPosY', 'X3.077_PeakPosY', 'X3.4015_PeakPosY', 'X3.6595_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
NextBestEst = RandomForestRegressor( n_estimators=800, max_depth=7, min_samples_leaf=4, n_jobs=10, oob_score=True, verbose=0 )
regCaReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, NextBestEst , 0.25 )
trainMSE, testMSE = regCaReduced.getErrors( iterations=20 )

print 'Ca best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

# Specify pH regressor
targetId = 'pH'
est = RandomForestRegressor( n_estimators=800, max_depth=None, min_samples_leaf=2, n_jobs=10, oob_score=True, verbose=0 )
regPhAll = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regAll.getErrors( iterations=20 )
print 'Ph All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))


featureSelection = ['id', 'EVI', 'X3.1065_PeakPosY', 'REF7', 'X3.718_PeakPosY', 'X3.4015_PeakPosY', 'X3.2235_PeakPosY', 'TMAP', 'BSAV', 'LSTD', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
reducedPhEst = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regPhReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, reducedPhEst, 0.25 )
trainMSE, testMSE = regPhReduced.getErrors( iterations=20 )
print 'pH best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))


# Specify SOC regressor use all features. Best result for features up to 25.
targetId = 'SOC'
est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regSOC = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regSOC.getErrors( iterations=20 )
print 'All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

featureSelection = ['id', 'X2.849_PeakPosY', 'X3.2715_PeakPosY', 'REF7', 'X3.6595_PeakPosY', 'X3.6005_PeakPosY', 'X3.4015_PeakPosY', 'X2.877_PeakPosY', 'X2.8225_PeakPosY', 'X3.1825_PeakPosY', 'X3.6085_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
reducedSOCEst = RandomForestRegressor( n_estimators=400, max_depth=None, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regSOCReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, reducedSOCEst, 0.25 )
trainMSE, testMSE = regSOCReduced.getErrors( iterations=20 )
print 'SOC best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f\t%s' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), clv.best_params_ )

# Specify Sand regressor use all features. Best result for features up to 25.
targetId = 'Sand'
est = RandomForestRegressor( n_estimators=800, max_depth=16, min_samples_leaf=1, n_jobs=10, oob_score=True, verbose=0 )
regSOC = ExploreRegressor.ExploreRegressor( df, targetId, est, 0.25 )
trainMSE, testMSE = regSOC.getErrors( iterations=20 )
print 'All features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ))

featureSelection = ['id', 'X3.077_PeakPosY', 'X3.1065_PeakPosY', 'X3.2715_PeakPosY', 'X3.298_PeakPosY', 'TMFI', 'TMAP', 'X3.0635_PeakPosY', 'X3.2555_PeakPosY', 'Ca', 'P', 'pH', 'SOC', 'Sand']
dfNew = df.ix()[:,featureSelection]
reducedSandEst = RandomForestRegressor( n_estimators=600, max_depth=4, min_samples_leaf=8, n_jobs=10, oob_score=True, verbose=0 )
regSandReduced = ExploreRegressor.ExploreRegressor( dfNew, targetId, reducedSandEst, 0.25 )
#clv = regSandReduced.gridSearch( verbose= 2 )
trainMSE, testMSE = regSandReduced.getErrors( iterations=20 )
print 'Sand best reduced features'
print 'meanTrainingMSE\tsdTrainingMSE\tmeanTestMSE\tsdTestMSE'
print '%f\t%f\t%f\t%f\t%s' % (np.mean( trainMSE ), np.std( trainMSE ), np.mean( testMSE ), np.std( testMSE ), clv.best_params_ )








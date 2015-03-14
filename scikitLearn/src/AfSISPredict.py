__author__ = 'carrillo'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import metrics
from sklearn.cross_validation import train_test_split

# create data frame
df = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingTransformed.csv',header=0)

#df = pd.read_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/trainingTransformedExpanded.csv',header=0)
df.head( 3 )
df.info()
df.describe()


label = df.iloc()[:,0]
input = df.iloc()[:,1:-6]
targets = df.iloc()[:,-6:]
target = targets.ix()[:,'P']


X_train, X_test, y_train, y_test = train_test_split( input, target, test_size=0.15, random_state=33)

# Learn random forest regressor
est = RandomForestRegressor(n_estimators=1000,
                            verbose=2,
                            n_jobs=10,
                            oob_score=True )

est.fit( X_train, y_train )

def printFeatureImportance( est, input ):
    fi = est.feature_importances_
    featureNames = list( input.columns.values)
    sortedIndex = sorted(range(len( fi )), key=lambda k: fi[k])
    for i in sortedIndex:
        print 'Feature %s \t Importance %f ' % ( featureNames[ i ], fi[ i ] )

printFeatureImportance( est, input )


est.oob_score_
est.oob_prediction_

# Plot prediction vs observed values
def plotPredictedVsObserved( est, input, target ):
    y_observed = target
    y_predicted = est.predict( input )
    plt.figure()
    plt.scatter( y_observed, y_predicted, c="k", label="data")
    plt.xlabel("observed")
    plt.ylabel("predicted")
    plt.legend()
    plt.show()


plotPredictedVsObserved( est, X_train, y_train )
plotPredictedVsObserved( est, X_test, y_test )

# Estimate errors via training and test set
def measure_performance(est, X, y ):
    y_pred=est.predict(X)
    print "Explained variance: {0:.5f}".format(metrics.explained_variance_score(y,y_pred)),"\n"
    print "Mean abs error: {0:.5f}".format(metrics.mean_absolute_error(y,y_pred)),"\n"
    print "Mean sqrt error: {0:.5f}".format(metrics.mean_squared_error(y,y_pred)),"\n"
    print "R2 score: {0:.5f}".format(metrics.r2_score(y,y_pred)),"\n"


measure_performance( est, X_train, y_train )
measure_performance( est, X_test, y_test )



# Estimates errors via cross validation
# scores = cross_validation.cross_val_score( est, input, target, cv=10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





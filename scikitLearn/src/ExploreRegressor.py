from sklearn import cross_validation
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import metrics
from sklearn.grid_search import GridSearchCV

__author__ = 'carrillo'


class ExploreRegressor:
    "Learns AfSIS regressor. Please provide the input X, the target parameter y and the regressor."

    def __init__( self, df, targetId ,reg, fractionTest ):
        # Extract X and Y from data frame and split into test and train sets.
        X = df.iloc()[:,1:-5]
        targets = df.iloc()[:,-5:]
        self.targetId = targetId
        y = targets.ix()[:,targetId]

        self.X = X
        self.y = y
        self.fractionTest = fractionTest
        self.reg = reg

    # Returns the Train MSE
    def getTestMSE( self ):
        y_predicted = self.reg.predict( self.X_test )
        return metrics.mean_squared_error( self.y_test, y_predicted )

    # Returns the Train MSE
    def getTrainMSE( self ):
        y_predicted = self.reg.predict( self.X_train )
        return metrics.mean_squared_error( self.y_train, y_predicted )

    def learn( self ):
        # Update training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.X, self.y, test_size=self.fractionTest )
        self.reg.fit( self.X_train, self.y_train )

    def reportCrossValidationError(self, cv=10 ):
        scores = cross_validation.cross_val_score( self.reg, self.X, self.y, cv=cv, scoring='mean_squared_error', n_jobs=10 )
        print("%s SQRT of MSE: %0.3f (+/- %0.3f)" % (self.targetId, np.sqrt( -scores.mean() ), np.sqrt( scores.std() * 2) ) )
        return scores

    def getLearningErrors(self, iterations, verbose=0 ):

        mseList = []
        for x in range(0, iterations ):
            self.learn()
            mse = self.getTestMSE()
            mseList.append( mse )

            if( verbose > 0 ):
                print 'Iteration %d current MSE %f' %( x, mse )

        return mseList

    # Returns lists of learning and testing errors
    def getErrors(self, iterations, verbose=0 ):
        mseTrain = []
        mseTest = []
        for x in range(0, iterations ):
            self.learn()
            testMse = self.getTestMSE()
            mseTest.append( testMse )

            trainMse = self.getTrainMSE()
            mseTrain.append( trainMse )


            if( verbose > 0 ):
                print 'Iteration %d current Test MSE %f' %( x, testMse )
                print 'Iteration %d current Train MSE %f' %( x, trainMse )

        return mseTest, mseTrain

    def gridSearch(self, tuned_parameters, cv=5, verbose=2 ):
        #tuned_parameters = [{'kernel': ['poly'], 'C': [9000,10000,11000], 'epsilon': [0.04,0.05,0.06], 'degree': [2], 'gamma': [0] }]

        clf = GridSearchCV( self.reg, tuned_parameters, cv=cv, verbose=verbose, n_jobs=10, scoring= 'mean_squared_error' )


        clf.fit( self.X, self.y )

        if( verbose > 0 ):
            print(clf.best_estimator_)
            #print(clf.best_params_)
            #print(clf.best_score_)

        self.reg = clf.best_estimator_
        return clf


    def getMeanLearningError(self, iterations ):
        return np.mean( self.getLearningErrors( iterations ) )

    # Return mean training and testing MSE
    def getMeanErros(self, iterations ):
        trainMse, testMse = self.getErrors( iterations )

        return np.mean( trainMse ), np.mean( testMse )

    def reportFeatureImportance( self, printImportance= True, plot=True ):

        importances = self.reg.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.reg.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        featureNames = list( self.X.columns.values)

        n = min( len( self.X.columns ), 10 )

        # Print the feature ranking
        if( printImportance ):
            print("Feature ranking:")

            for f in range( n ):
                print("%d. feature %s (%f)" % (f + 1, featureNames[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        if( plot ):
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(10), importances[indices[0:n]], color="r", yerr=std[indices[0:n]], align="center")
            #plt.bar( range(10), height=importances[indices[0:10]] )
            plt.xticks(range(n), indices)
            plt.xlim([-1, n])
            plt.show()

        return [ featureNames[i] for i in indices ]

    # Learn the best features by sampling important features from different learning iterations.
    def getImportantFeatures(self, featuresToRetrieve = 10, learningIterations = 10, featuresToCollectPerIteration=10, verbose=1 ):

        if( verbose > 0 ):
            print 'Learning important features.'

        featureDict = dict()
        for i in range(0,learningIterations):
            self.learn()
            importantFeatures = self.reportFeatureImportance( printImportance=False, plot=False )[0:featuresToCollectPerIteration]
            for j in range(0,featuresToCollectPerIteration ):
                if( importantFeatures[ j ] in featureDict ):
                    featureDict[ importantFeatures[ j ] ] += 1
                else:
                    featureDict[ importantFeatures[ j ] ] = 1

        out = ["id"] + sorted(featureDict, key=featureDict.get, reverse=True)[ 0 : ( featuresToRetrieve ) ] + ["Ca","P","pH","SOC","Sand"]

        if( verbose > 0 ):
            print out
            print 'Learning important features. Done'

        return out


    # Plot prediction vs observed values
    def plotPredictedVsObserved( self, input, target ):
        y_observed = target
        y_predicted = self.reg.predict( input )

        plt.figure()
        plt.scatter( y_observed, y_predicted, c="k", label="data")
        plt.xlabel("observed")
        plt.ylabel("predicted")
        plt.legend()
        plt.show()

    # Plot training error
    def plotTrainingError( self ):
        self.plotPredictedVsObserved( self.X_train, self.y_train )

    # Plot training error
    def plotTestError( self ):
        self.plotPredictedVsObserved( self.X_test, self.y_test )

    # Estimate errors via training and test set
    def reportPerformance( self, X, y ):
        y_pred=self.reg.predict(X)
        print "Explained variance: {0:.5f}".format(metrics.explained_variance_score(y,y_pred)),"\n"
        print "Mean abs error: {0:.5f}".format(metrics.mean_absolute_error(y,y_pred)),"\n"
        print "Mean sqrt error: {0:.5f}".format(metrics.mean_squared_error(y,y_pred)),"\n"
        print "R2 score: {0:.5f}".format(metrics.r2_score(y,y_pred)),"\n"

    def reportTrainingPerformance( self ):
        self.reportPerformance( self.X_train, self.y_train )

    def reportTestPerformance( self ):
        self.reportPerformance( self.X_test, self.y_test )

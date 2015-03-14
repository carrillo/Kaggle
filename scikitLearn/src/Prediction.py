from pandas import Series
import matplotlib.pyplot as plt
import pandas as pd

__author__ = 'carrillo'


class Prediction:
    "Learns AfSIS regressor for prediction. Please provide the input X, the target parameter y and the regressor."

    def __init__( self, train, test, est, featureSelection, targetId ):
        self.train = train;
        self.test = test;

        print "Nr of features in train and test: %d and %d" %( len( self.train.columns ), len( self.test.columns ) )

        self.est = est
        self.featureSelection = featureSelection
        self.targetId = targetId




    def learn(self ):
        print 'Learning %s data' % (self.targetId)

        if self.featureSelection is None:
            df = self.train
        else:
            df = self.train.ix()[:, self.featureSelection ]



        X = df.iloc()[:,1:-5]

        y = df.ix()[ :,self.targetId ]

        self.est.fit( X, y )




        print 'Learning %s data. Done.' % (self.targetId)


    def getTargetSpecificTestSet(self, featureSelection ):
        if featureSelection is None:
            df = self.test
        else:
            df = self.test.ix()[ :, featureSelection ]
        return df

    def predict(self):
        print 'Predicting %s data' % (self.targetId)
        if self.featureSelection is None:
            df = self.test
        else:
            testfs = self.featureSelection
            testfs.remove('Ca')
            testfs.remove('P')
            testfs.remove('pH')
            testfs.remove('SOC')
            testfs.remove('Sand')
            df = self.test.ix()[ :, testfs ]


        id = df.ix()[:,'PIDN']
        X_test = df.iloc()[:,1::]

        y_pred = self.est.predict( X_test )


        d = { self.targetId : y_pred, 'PIDN' : id  }

        print 'Predicting %s data. Done.' % (self.targetId)

        return pd.DataFrame( d )









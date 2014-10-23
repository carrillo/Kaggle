__author__ = 'carrillo'

####################################
# Selects a subset of training examples to match the test set.
# Performs nearest neighbour search within the training set for each test set.
####################################

import pandas as pd
from sklearn.neighbors import KDTree

# Opens the data and replaces Depth nominal to numeric value if necessary.
def getData( fileName ):
    data = pd.read_csv( fileName ,header=0)
    data.replace( to_replace='Topsoil', value=0, inplace=True )
    data.replace( to_replace='Subsoil', value=1, inplace=True )
    return data

# Drops ID values in test and training and target values in training sets.
def extractFeatures( set ):
    a = set.drop("PIDN",1)
    if "P" in set:
        a = a.drop(["Ca","P","pH","SOC","Sand"],1)

    return a

# Finds the n nearest neighbors of the training set for each test sample. Returns the indices of the nearest neighbor set.
def findNearestNeighborsOfTestInTrain( test, train, n_neighbors ):

    testFeatures = extractFeatures( test )
    trainFeatures = extractFeatures( train )

    kdt = KDTree( trainFeatures, leaf_size=30, metric='euclidean')
    trainIndices = kdt.query( testFeatures, k=n_neighbors, return_distance=False)

    indexSet = set( trainIndices.flatten().tolist() )

    return( indexSet )


n_neighbors = 10 # Define the number of neighbors to be retrieved for each test set.

# Load data.
id = "DecaySubtractedPrincipalComponents"
train = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/training' + id + '.csv')
test = getData('/Users/carrillo/workspace/Kaggle/resources/AfSIS/test' + id + '.csv')

# Get indices of matching training samples and subset training set
trainIndex = findNearestNeighborsOfTestInTrain( test, train, n_neighbors )
trainSubset = train.iloc()[ list( trainIndex) ,:]

# Write to file. 
print 'Reduced %d training samples to %d by matching with test set.' % ( len( train ), len( trainSubset ) )
trainSubset.to_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/training' + id + '_matchedWithTestSet.csv',header=True,index=False)
test.to_csv('/Users/carrillo/workspace/Kaggle/resources/AfSIS/test' + id + '_DepthNumeric.csv',header=True,index=False)



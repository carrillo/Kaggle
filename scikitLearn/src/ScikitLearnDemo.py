__author__ = 'carrillo'

from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

# Looking at the data
print( digits.data )
digits.target

# Training a SVM classifier with all but the last data sample
from sklearn import svm
clf = svm.SVC( gamma=0.001, C=100. )
clf.fit(digits.data[:-1], digits.target[:-1])

# test on last digit image
clf.predict(digits.data[-1])










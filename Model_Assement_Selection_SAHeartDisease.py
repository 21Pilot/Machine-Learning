# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/10/7
# Frequentist Machine Learning_Project3

# Set Up
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Data consturction
# Identities follow the textbook instruction 
X = np.random.normal(0,1,[50, 5000])
Y = np.concatenate([np.zeros(25), np.ones(25)])
np.random.shuffle(Y)

# Incorrect Cross Validation # Textbook
CV_correct = []
# Screening the subset of predictors with strong correlation with class labels.
X_new = preprocessing.MinMaxScaler().fit_transform(X)
X_new = SelectKBest(chi2, k = 100).fit_transform(X_new, Y)

# Multivariate classifier with the selected subset of predictors
KNN = KNeighborsClassifier(n_neighbors=1)
# Cross Validation for tuning unknown parameters/ estimate error
rkf = RepeatedKFold(n_splits=5, n_repeats=50)

for train_index, test_index in rkf.split(X_):
    X_train, X_test = X_new[train_index], X_new[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    KNN.fit(X_train, Y_train)
    CV_correct.append(1-KNN.score(X_test, Y_test))
print("Average CV Error Rate:", np.array(CV_correct).mean())

X = pd.DataFrame(data=X)
Y = pd.DataFrame(data=Y)


# Correct Cross Validation # Textbook
CV_correct = []

# Cross Validation for tuning unknown parameters/ estimate error
rkf = RepeatedKFold(n_splits=5, n_repeats=50)

# Division following steps (1,2,3)
for train_index, test_index in rkf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

# Screening the subset of predictors with strong correlation with class labels.
    X_new = preprocessing.MinMaxScaler().fit_transform(X_train)
    X_new = pd.DataFrame(data=X_new)

    kbest = SelectKBest(chi2, k=100)
    kbest.fit_transform(X_new, np.ravel(Y_train))

    best_features = kbest.get_support()
    X_new = X_new.iloc[:, best_features]

    # Multivariate classifier with the selected subset of predictors
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_new, np.ravel(Y_train))

    CV_correct.append(1-neigh.score(X_test.iloc[:, best_features], np.ravel(Y_test)))
print("Average Cross Validation Error Rate:", np.array(CV_correct).mean()) 

# Multivariate Classifier with KNN system for classifying IRIS data set
# Set Up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics 

iris = load_iris( )
print( iris.data.shape , iris.feature_names , iris.target.shape , iris.target_names , 
      type( iris ) , type( iris.data ) )

#  https://medium.com/analytics-vidhya/exploration-of-iris-dataset-using-scikit-learn-part-1-8ac5604937f8
# IRIS data processing 
df = pd.DataFrame(data= np.c_[ iris[ 'data' ] , iris[ 'target' ] ] , 
                  columns = iris[ 'feature_names' ] + [ 'Species' ] ) 

#  feature matrix , resp vector
# (150, 4) ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 
# 'petal width (cm)'] (150,) 
# ['setosa' 'versicolor' 'virginica'] 
# <class 'sklearn.utils.Bunch'> <class 'numpy.ndarray'>
X , y = iris.data , iris.target  

# Split the data to test/ train 
X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.4 , random_state = 4 )
print ( X_train.shape , X_test.shape , y_train.shape , y_test.shape , df.shape , df.columns , df.describe( ) )

# Multivariate KNN Classifier
# With KNN Estimator K set as 5, the data goes through the model training (fit), then its accuracy is tested in the range of 1 ~ 26
# Estimator for K=5
knn = KNeighborsClassifier( n_neighbors = 5 )  
#  Model training
knn.fit( X_train , y_train )  
y_pred = knn.predict( X_test )
#  Test accuracy of K=5 in range of 1-26
print( metrics.accuracy_score( y_test , y_pred ) )  
k_range = range( 1 , 26 )

# Scoring system 
# 5 scores of K-Fold splitting cross validation at the end of the loop
scores = [ ]
for k in k_range :
    knn = KNeighborsClassifier( n_neighbors = k )
    knn.fit( X_train , y_train )
    y_pred = knn.predict( X_test )
    scores.append(metrics.accuracy_score( y_test , y_pred ) )
print( X_test[ : 5 ] , y_test[ : 5 ] )

# Result
plt.plot( k_range , scores )
plt.xlabel( 'Values of K' )
plt.ylabel( 'Accuracy' )
plt.show( )  
# Block = False
# Optimal K = 11
knn = KNeighborsClassifier( n_neighbors = 11 )  
# Model training
knn.fit( X , y )  
# [3,5,4,2] = arbitrary measurement to find out what model would predict
# Predict out-of-sample data
chk = knn.predict( [ [ 3 , 5 , 4 , 2 ] ] ) 
print( chk , metrics.accuracy_score( y_test , knn.predict( X_test ) ) )

# End of Incorrect CV

# Correct CV IRIS
# Set Up
from sklearn.model_selection import KFold , cross_val_score , StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import svm

# Cross Validation implicating KFold with 5 splits.
# https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869
# https://scikit-learn.org/stable/modules/cross_validation.html
clf = svm.SVC( kernel='linear' , C = 1 )
kf = KFold( n_splits = 5 )

for train_index , test_index in kf.split( range( len( X ) ) ) :
  print("TRAIN :" , train_index[ : : 10 ] , "TEST :" , test_index[ : : 10 ] )
  X_train , X_test = X[ train_index ] , X[ test_index ]
  y_train , y_test = y[ train_index ] , y[ test_index ]
  clf.fit( X_train , y_train )
  y_pred = clf.predict( X_test )

  #  https://stackoverflow.com/questions/39826538/cross-validation-with-particular-dataset-lists-with-python
  #  https://stackoverflow.com/questions/42075986/scikitlearn-score-dataset-after-cross-validation
  score = accuracy_score( y_test , y_pred )
  print( score )  
  skf = StratifiedKFold(n_splits = 5 )  
  scores = cross_val_score( clf , X , y , cv = skf )
  print( scores , scores.mean( ) )
  
# Correct CV SAHeartDisease
# Set Up
from sklearn.model_selection import KFold , cross_val_score , StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import svm

# Load Data 
df2 = pd.read_csv( "South African Heart Disease.txt" , header = 0 )

# Treating 'famhist' as dummy variable 
df2[ 'famhist' ] = pd.get_dummies( df2[ 'famhist' ] )[ 'Present' ]
target = 'chd'
features = [ c for c in df2.columns if ( c[ 0 ] not in [ 'r' , 'c' ] ) ]
df2.shape , df2.columns , df2[ features + [ target ] ][ : 2 ]

# Cross Validation implicating KFold with 5 splits.
[df2[ c ].update( ( df2[ c ] - df2[ c ].min( ) ) / ( df2[ c ].max( ) - df2[ c ].min( ) ) ) 
for c in features if c not in [ target , 'famhist' ] ]
X2 , y2 = df2[ features ].to_numpy( ) , df2[ target ].to_numpy( )
clf2 = svm.SVC( kernel = 'linear' , C = 1 )
kf2 = KFold( n_splits = 5 )

# https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869
for train_index , test_index in kf2.split( range( len( X2 ) ) ) :
  print( "TRAIN :" , train_index[ : : 10 ] , "TEST :" , test_index[: : 10 ] )
  X_train , X_test = X2[ train_index ] , X2[ test_index ]
  y_train , y_test = y2[ train_index ] , y2[ test_index ]
  clf2.fit( X_train , y_train )
  y_pred = clf2.predict( X_test )
  score = accuracy_score( y_test , y_pred )

#  https://stackoverflow.com/questions/39826538/cross-validation-with-particular-dataset-lists-with-python
#  https://stackoverflow.com/questions/42075986/scikitlearn-score-dataset-after-cross-validation
print( score )  
skf = StratifiedKFold(n_splits = 5 )  
scores = cross_val_score( clf , X2 , y2 , cv = skf )
print( scores , scores.mean( ) )

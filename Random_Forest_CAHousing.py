# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/10/17
# Frequentist Machine Learning_Project5

# Data 
# Data from Carngie-Mellon SataLib repositry : as well as from skitlearn datasets.
# "California Hosing.txt

# Set Up 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# California housing data load 
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing() 
# Feature name list 
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']
# DataFrame
X = pd.DataFrame(housing.data, columns = housing.feature_names)
y = pd.DataFrame(housing.target, columns = ['MedPrice'])
X.head()
y.head()

# Cheacking for anamoly 
y.hist(bins=100, figsize=(10,10))
plt.show()
df = pd.concat([X, y], axis=1)
# There exits anamoly at the end of the graph
# Such anamoly is deleted 
df=df.loc[df['MedPrice']<4.5,:]
df.shape

# Explanatory variable and Dependent variable separated  
y = df['MedPrice']
df.drop('MedPrice',axis=1,inplace=True)
X = df
# Test and train test splited 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=1)

# LinearRegression 
# R Squared score 
# Mean Absolute Error 
regressor= LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

r2=r2_score(y_test,y_pred)
print('the R squared of the linear regression is:', r2)
print('the mean absolute error is :', mean_absolute_error(y_test, y_pred))

# Random Forest 
# R Squared score 
# Mean Absolute Error 
regressor = RandomForestRegressor(n_estimators=50, min_samples_split = 6)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

r2=r2_score(y_test,y_pred)
print('the R squared of the linear regression is:', r2)
print('the mean absolute error is :',mean_absolute_error(y_test, y_pred))

# Feature Imporatnce using only Random Forest 
# Numerical value of imporatnce of each feature is presented
# This is done just to cross check later in the analyzation 
importances = regressor.feature_importances_
print('the importance of each feature is as ordered :',importances)
plt.figure()
plt.title("Feature importance - California Housing")
plt.xticks(fontsize=8, rotation=60)
plt.bar(col, importances)

# Check Random Forest with different Tree qunatity. 
# Two random forest are shown, with m =2 and m =6 
ls = list(range(100))
trees = ls[10::20]
ls = list(range(500))
trees.extend(ls[100::100])

# Score1-RF (mean_absolute_error) with minimum 2 samples split 
scores1 = []
for tree in trees:
    regressor = RandomForestRegressor(n_estimators=tree, min_samples_split = 2)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores1.append(mean_absolute_error(y_test, y_pred))
    # print(tree, 'trees done')
importances1 = regressor.feature_importances_

# Score2-RF (mean_absolute_error) with minimum 6 samples split 
scores2 = []
for tree in trees:
    regressor = RandomForestRegressor(n_estimators=tree, min_samples_split = 6)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores2.append(mean_absolute_error(y_test, y_pred))
    # print(tree, 'trees done')
importances2 = regressor.feature_importances_

# Two gradient boosted models have interaction depths of 4 and 6 

# Score3-GB (mean_absolute_error) with maximum depth of 4
scores3 = []
for tree in trees:
    regressor = GradientBoostingRegressor(n_estimators=tree, max_depth = 4)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores3.append(mean_absolute_error(y_test, y_pred))
    print(tree, 'trees done')
importances3 = regressor.feature_importances_

# Score4-GB (mean_absolute_error) with maximum depth of 6
scores4 = []
for tree in trees:
    regressor = GradientBoostingRegressor(n_estimators=tree, max_depth = 6)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores4.append(mean_absolute_error(y_test, y_pred))
    print(tree, 'trees done')
importances4 = regressor.feature_importances_

# Figure Plot:  Random forest compared to gradient boositng on the California Housing data. 
# The curves represent mean absolute error on the test data as a function of the number of trees in the model'
# Figure 15.3 
plt.plot(trees, scores1, marker='o', label='RF, m=2')
plt.plot(trees, scores2, marker='o', label='RF, m=6')
plt.plot(trees, scores3, marker='o', label='GBM, depth=4')
plt.plot(trees, scores4, marker='o', label='GBM, depth=6')
plt.xlabel('Number of Trees')
plt.ylabel('Test Average Absolute Error')
plt.title('California Housing Data')
plt.legend()

# Feature importance calculated using only Random Forest
plt.figure()
plt.title("Feature importances-Random Forest")
plt.xticks(fontsize=8, rotation=60)
plt.bar(col, importances2)

# Feature importance calculated using only Gradient Boost 
plt.figure()
plt.title("Feature importances - Gradient Boost")
plt.xticks(fontsize=8, rotation=60)
plt.bar(col, importances4)

# Additional Example_IRIS
# Additional Set Up 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Iris data load
# Iris data process 
df=pd.read_csv("iris.csv")
print(df.shape)
X = df.loc[:,'sepal.length':'petal.width']
y = df.loc[:,'variety']
le = LabelEncoder()
T = le.fit_transform(y) 

# Test and Train data splited
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
# Iris data feature listed 
col = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

# narrower range 
trees = list(range(1,30))

# Replicate the random forest/ gradient boost analysis 
# Random Forest
# NOTE the score is accuracy score, not mean abs error 

# Score1-RF accuracy score with minimum 2 samples split 
scores1 = []
for tree in trees:
    Classifier = RandomForestClassifier(n_estimators=tree, min_samples_split = 2)
    Classifier.fit(X_train, y_train)
    y_pred = Classifier.predict(X_test)
    scores1.append(accuracy_score(y_test, y_pred))
    # print(tree, 'trees done')
importances1 = Classifier.feature_importances_

# Score2-RF accuracy score with minimum 6 samples split 
scores2 = []
for tree in trees:
    Classifier = RandomForestClassifier(n_estimators=tree, min_samples_split = 6)
    Classifier.fit(X_train, y_train)
    y_pred = Classifier.predict(X_test)
    scores2.append(accuracy_score(y_test, y_pred))
    # print(tree, 'trees done')
importances2 = Classifier.feature_importances_

# Gradient Boost 
# Score3-GB accuracy score with maximum depth of 4
scores3 = []
for tree in trees:
    Classifier = GradientBoostingClassifier(n_estimators=tree, max_depth = 4)
    Classifier.fit(X_train, y_train)
    y_pred = Classifier.predict(X_test)
    scores3.append(accuracy_score(y_test, y_pred))
    # print(tree, 'trees done')
importances3 = Classifier.feature_importances_

# Score4-GB accuracy score with maximum depth of 6
scores4 = []
for tree in trees:
    Classifier = GradientBoostingClassifier(n_estimators=tree, max_depth = 6)
    Classifier.fit(X_train, y_train)
    y_pred = Classifier.predict(X_test)
    scores4.append(accuracy_score(y_test, y_pred))
    # print(tree, 'trees done')
importances4 = Classifier.feature_importances_

# IRIS data set has only 4 features, with easy interpretation. 
# This aspect resulted in a very high accuracy score
plt.plot(trees, scores1, marker='o', label='RF, m=2')
plt.plot(trees, scores2, marker='o', label='RF, m=6')
plt.plot(trees, scores3, marker='o', label='GBM, depth=4')
plt.plot(trees, scores4, marker='o', label='GBM, depth=6')
plt.xlabel('Number of Trees')
plt.ylabel('Test Accuracy Score')
plt.title('IRIS DATA')
plt.legend()

# Feature importance calculated using only Random Forest
plt.figure()
plt.title("Feature importances - Random Forest")
plt.xticks(fontsize=8, rotation=60)
plt.bar(col, importances2)

# Feature importance calculated using only Gradient Boost 
plt.figure()
plt.title("Feature importances - Gradient Boost ")
plt.xticks(fontsize=8, rotation=60)
plt.bar(col, importances4)

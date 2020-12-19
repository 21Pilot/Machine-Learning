# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/10/17
# Frequentist Machine Learning_Project4

# Data 
# Data from Carngie-Mellon SataLib repositry : as well as from skitlearn datasets.
# "California Hosing.txt"

# Set Up
# XGBoost 
import xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance

# Skleran
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import model_selection
from sklearn.model_selection import train_test_split 

# Usual friends.. 
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import numpy as np

# Data Load
# California housing data load 
housing = fetch_california_housing()
col_names = housing.feature_names
# DF
X_df = pd.DataFrame(housing.data, columns = col_names)
y = housing.target
# Data split 
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, random_state = 20)

# Absolute average erros in respect to iteration M
# Mean Absolute Error range for plot 
# check textbook 
mean_abs_err = range(0,30)

# Initiate train and test MAE
train_mae = []
test_mae = []

# Train evaluation over the data set 
for evaluation in mean_abs_err : 
  model =XGBRegressor(objective ="reg:squarederror",
                      learning_rate = 0.1, n_estimators = evaluation)
  model.fit(X_train, y_train)

  # Mean Sqaured Error  
  y_test_pred = model.predict(X_test)
  y_train_pred = model.predict(X_train)

  train_mae.append(mean_absolute_error(y_train,y_train_pred)) 
  test_mae.append(mean_absolute_error(y_test, y_test_pred))
  print('Mean Absolute Error_train:', mean_absolute_error(y_train, y_train_pred))
  print('Mean Absolute Error_test:', mean_absolute_error(y_test, y_test_pred))

  # print('Mean Absolute Error:', mean_abs_err)

# Plot 
plt.plot(mean_abs_err, train_mae)
plt.plot(mean_abs_err, test_mae)
plt.title("Training and Test Mean Absolute Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Mean Absolute Error")
plt.legend(["Train mean absolute error", "Test mean absolute error"])
plt.show()

# Relative importance of the predictor for the California housing data
# plot_importance 
xgboost.plot_importance(model)
plt.yticks(range(8), col_names)
plt.title ('Feature importance')
plt.show()

# Additional Example 
# SAHeartDisease

df = pd.read_csv( "South African Heart Disease.txt")
df['famhist'] = pd.get_dummies(df['famhist'])['Present']
target = 'chd'
features = ['sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age']
df[features + [target]].head()
dfCol = features + [target]
print (dfCol, df.columns.values)
df.columns.values 
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state = 20)

parameters = {
    'nthread':                  [4],
    'learning_rate':            [0.1, 0.3, 0.5],
    'objective':                 ['reg:linear'],
    'n_estimators':             [500],
    'lambda': [.1, 1, 5],
    'alpha': [0, 3,5],
    'eval_metric': ['mae']
} 

# Mean Absolute Error of train and test vs Number of iteration
# Mean Absolute Error range for plot 
# check textbook 
mean_abs_err = range(0,30)

# Initiate train and test MAE
train_mae = []
test_mae = []

# Train evaluation over the data set 
for evaluation in mean_abs_err : 
  model_SA =XGBRegressor(objective ="reg:squarederror",
                      learning_rate = 0.1, n_estimators = evaluation)
  model_SA .fit(X_train, y_train)

  # Mean Sqaured Error  
  y_test_pred = model_SA .predict(X_test)
  y_train_pred = model_SA .predict(X_train)
  train_mae.append(mean_absolute_error(y_train,y_train_pred)) 
  test_mae.append(mean_absolute_error(y_test, y_test_pred))
  print('Mean Absolute Error_train:', mean_absolute_error(y_train, y_train_pred))
  print('Mean Absolute Error_test:', mean_absolute_error(y_test, y_test_pred))
  
# Plot Training and Test Mean Absolute Error
plt.plot(mean_abs_err, train_mae)
plt.plot(mean_abs_err, test_mae)
plt.title("Training and Test Mean Absolute Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Mean Absolute Error")
plt.legend(["Train mean absolute error", "Test mean absolute error"])
plt.show()

# Feature Importance Plot SAHD
# plot_importance 
xgboost.plot_importance(model_SA)
plt.yticks(range(len(dfCol)),dfCol)
plt.title('Feature importance')
plt.show()


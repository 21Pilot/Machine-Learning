# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/9/11
# Frequentist Machine Learning_Project1

# Data 
# Data from 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
# "prostate.data.txt"

# Set Up 
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# The data has both INT and FLOAT data, so the two were recognized.
# For analyzation convenience, intercept : column of 1 was added
# INT - age/ siv/ gleason/ pgg45
# FLOAT - lcavol/ lweight/ lbph/ lcp/ lpsa

rows = []
with open("prostate.data.txt","r") as file:
  lines = file.readlines()
  columns = lines[0].split('\t')
  columns[0] = 'index'
  columns[-1] = columns[-1][:-1]
  for line in lines[1:]:
      row = line.split('\t')
      rows.append(row)
      
     
data = pd.DataFrame(rows, columns=columns)
# Last colume (train/test indicator) drop
data.drop(['index','train'],axis=1,inplace = True)

# INT data and FLOAT data recognized
data.lcavol = data.lcavol.astype('float')
data.lweight = data.lweight.astype('float')
data.age = data.age.astype('int')
data.lbph = data.lbph.astype('float')
data.svi = data.svi.astype('int')
data.lcp = data.lcp.astype('float')
data.gleason = data.gleason.astype('int')
data.pgg45 = data.pgg45.astype('int')
data.lpsa = data.lpsa.astype('float')

# Columns of 1 as intercept 
data['intercept'] = 1
data.head()

# Check shape ( 97 rows, 10 columns)
data.shape

# Divide data into roughly 80% train, 10% validation, 10% test.
X = data.drop(['lpsa'],axis=1)
y = data['lpsa']
# Train 80%/ Test 10%/ Val 10%/ sklearn train_test_split
X,X_test, y, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

# Check shape
X_train.shape, X_val.shape, X_test.shape
y_train.shape, y_val.shape, y_test.shape

# Linear Regression 
# Linear regression here was achieved through strickly following the textbook formulas.
# Linear regression performed on Training dataset.
# The parameter set Beta is estimated from training dataset, using MAT form formulas.
 
def linear_regression(X_train,y_train): 
  # X_transpose * X / dot product
  # X_transpose = X_train.T 
  XTX = np.dot(X_train.T, X_train)

  # Inv of (X_trainspose * X) / linear algebra
  XTX_inv = np.linalg.inv(XTX)

  # (Inv of (X_trainspose * X)) * (x_transpose) * y / dot product 
  # y = y_train
  # Equ 3.6
  betas = np.dot(np.dot ( XTX_inv,X_train.T),y_train)

  return betas

# Betas
betas = linear_regression(X_train, y_train)
betas

# The minimized Mean Squared Error is calcualted based on estimated parameter set Beta.
# The error is yield betweent the Predicted outcome based on input test dataset and estimated beta and output test dataset.

# Predicted y = y_hat
y_pred = np.dot(X_test, betas)
# MSE between predicted y and test y 
print( "Report Mean Squared Error : ", mean_squared_error(y_pred, y_test))

# Correlation Table 
print('Table :correlations of predictors in the prostate cancer data - symmetrical.')
X_train.corr()

# Z Score Table
# v_j = Inv of (X_transpose * X ) /linear algebra/ dot product 
XTXI = np.linalg.inv(np.dot(X_train.T,X_train))
# Sqrt_v_j in range of v_j
sqrt_v = np.sqrt([XTXI[i][i] for i in range(len(XTXI))])
# Parameter for sigma
N = len(X_test)
P = len(X_test.columns)-1
# Sigma from equ 3.8/ (y_test - y_pred)^2 = dot product of (y_test-y_pred)
sigma = np.sqrt (np.dot((y_test - y_pred), (y_test - y_pred))*(1/(N-P-1)))
# Standard Error = sigma * sqrt_v_j
std_err = sigma * sqrt_v 
# z_j equ. 3.12
Z = np.divide ( betas, std_err)

print('Table 3.2: Z score table of the prostate cancer data.')
Z_table = pd.DataFrame()
Z_table["Term"] = X_test.columns
Z_table["Std. err"] = std_err
Z_table["Z Values"] = Z
Z_table

# Ridge Linear Regression 
# The Ridge linear regression here was achieved through strickly following the textbook formulas.
# The parameter set Beta is estimated from by ridge regression without the intercept
def ridge(X_train, y_train, alpha): 
    n, m = X_train.shape
    #Identity matrix
    I = np.identity(m)
    # X_transpose * X / dot product
    XTX = np.dot(X_train.T,X_train)
    # X_transpose * X + alpha *I
    XTXA = XTX + alpha * I
    # Inv of (X_trasnpose * X + alpha *I) / linear algebra
    XTXA_inv = np.linalg.inv(XTXA)
    # Equ 3.44 (Inv of (X_transpose * X)) * X_transpose * y / dot product
    betas = np.dot(np.dot(XTXA_inv, X_train.T), y_train)
    return betas
betas = ridge(X_train,y_train,1)
betas 

# Calculate best Alpha by cross validation 
# Cross Val over lambda range of 0 ~ 5, step = 0.25
alphas = np.arange(0.0, 5.0, 0.25)
for alpha in alphas:
    betas = ridge(X_train,y_train,alpha)
    y_pred = np.dot(X_val,betas)
    mse = mean_squared_error(y_pred, y_val)
    print(f"Alpha = {alpha}, MSE = {mse}")
labels= X_train.columns

# Ridge Plot 3.8 
# First Version 
# Some lambda values reccomended from scikitlearn
for lam in [1e-15, 1e-10, 1e-5, .1, 1,1e-4, 1e-2,1e-3, 1, 5, 10]:
  # Use eq 3.44 to find the betas
  # β̂ ridge = (X T X + λI) −1 X T y
  identity = np.identity(X_train.shape[1])
  categories = list(X_train.columns.values)
  beta_ridge_hat = np.linalg.inv((X_train.T.dot(X_train) + lam*(identity))).dot(X_train.T.dot(y_train))
  # ŷ = X β̂
  r_y_hat = X_val.dot(beta_ridge_hat)

lams = np.logspace(-8, 4, 1000)
X = []
Y = []
for lam in lams:
  betaRidge_hat = np.linalg.inv((X_train.T.dot(X_train) + lam*(identity))).dot(X_train.T).dot(y_train)
  # df(λ)= tr[X(X T X + λI) −1 X T ] eq 3.50
  # DOF = np.matrix(r_x_train.dot(np.linalg.inv((r_x_train.T.dot(r_x_train) + lam*(identity))).dot(r_x_train.T)))
  X.append(lam)
  Y.append(betaRidge_hat)
plt.figure
plt.xscale("log")

for i in range(0,8):
  plt.plot(X, list(map(list, zip(*Y)))[i], label = categories[i] )
plt.title("Profiles of ridge coefficients")
plt.xlabel('λ')
plt.ylabel('Coefficients') 
plt.legend()
ymin, ymax = plt.ylim()
plt.vlines(bestLambda, ymin, ymax, linestyle='dashed', colors='red')
plt.show()

# Ridge Plot 3.8 
# Second Version 
alphas = np.arange(0.0, 5.0, 0.1)
coefs = []
for a in alphas:
    betas = ridge(X_train,y_train,a)
    coefs.append(betas)
coefs = np.array(pd.DataFrame(coefs).T)
ax = plt.gca()

# Scale 
for i in range(len(coefs)):
    ax.plot(alphas, coefs[i],label=X_train.columns[i])
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])

# Plot figure 3.8 - Ridge coefficient. Ridge
plt.xlabel('alpha : Lambda')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.legend()
plt.show()

# Lasso Regression
# Lasso Plot
alphas = np.arange(0.0, 5.0, 0.1)
coefs = []
for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train.drop(['intercept'],axis=1),y_train)
    coefs.append(lasso.coef_)

coefs = np.array(pd.DataFrame(coefs).T)
ax = plt.gca()
# print(coefs)
for i in range(len(coefs)):
    ax.plot(alphas, coefs[i],label=X_train.columns[i])
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])

# Plot figure 3.10 - Ridge coefficient.Lasso 
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/11/20
# Frequentist Machine Learning_Project7

# Data 
# Data Source : http://www2.informatik.uni-freiburg.de/~cziegler/BX/ from https://gist.github.com/entaroadun/1653794
# "BX-Book-Ratings.csv"

# Set Up
# Install Surprise
!pip install scikit-surprise

import pandas as pd
import numpy as np
from surprise import NMF
from surprise.model_selection.split import train_test_split
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset
from surprise.model_selection import GridSearchCV

df=pd.read_csv('BX-Book-Ratings.csv',sep=';',encoding='unicode_escape')
df.isnull().sum()
df.info()

# Changing str categorical data to str quantitative data to minimize the trianing time 
df.ISBN = pd.factorize(df.ISBN)[0]
df['Book-Rating'].unique()
# To pervent 'zero division', the ratings of '0' are excluded. 
df=df.loc[df['Book-Rating']!=0]

# Filtering to faciliate training time 
# Book Filter
min_book_ratings = 20
filter_books = df['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

# User Rating Filter 
min_user_ratings = 20
filter_users = df['User-ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist() 

# Filtered results 
df_new = df[(df['ISBN'].isin(filter_books)) & (df['User-ID'].isin(filter_users))]

# NVM
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_new[['User-ID', 'ISBN', 'Book-Rating']], reader)
algo = NMF()
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)

# Parameter grid for finding best value hyperparameter
param_grid = {'n_factors': [3,5,10,15], 'n_epochs':[10,30,50],'lr_bu': [0.002, 0.005],
              'reg_pu': [0.4, 0.6]}
gs = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# Best RMSE score
print(gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# RMSE from system with foldings
# Using RMSE from system 
# 3 folds 
algo = NMF(n_factors= 15, n_epochs=50, lr_bu= 0.002, reg_pu= 0.6)
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True

# Using RMSe equation 7 # Textbook
# Fit and test the model with 0.2/0.8 split 
train, test = train_test_split(data, test_size = 0.2)
algo.fit(train)

# Predictions
predictions = algo.test(test)
prediction_array = []
for prediction in predictions :
  uid = int(prediction.uid)
  iid = int(prediction.iid)
  r_ui = int(prediction.r_ui)
  est = float(prediction.est)
  prediction_array.append([uid, iid, r_ui, est])
pred = np.array(prediction_array)

# Using RMSE equation 7
rmse = np.sqrt(np.mean((pred[:,2] - pred[:,3]) ** 2))

# Accuracy
est_rounded = pred[:,3].astype(int)
accuracy = np.sum(pred[:,2] != est_rounded) / pred.shape[0]

# Print out test metrics
print(f'RMSE: {rmse}; Accuracy: {accuracy}')

# str to dataframe for readability
predict = pd.DataFrame(predictions)

# Filter predictions and print only quality recommendation
# quality : est>4.9
recommendations = predict[predict['est'] > 4.9] 
print(recommendations)

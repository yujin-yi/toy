## Load Package

import pandas as pd
import numpy as np
import random
import time
from math import *

from itertools import combinations

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

## Load Binary Dataset
dataset = load_breast_cancer(as_frame=True)

## Split Dataset to get data trained
real_X = dataset['data']
real_y = dataset['target']

## Make Noise y data to use Mis-Labeled Classification Approach
rn = real_X.shape[0]
rate = 0.1
fake_rn = random.sample(range(rn), round(rn * rate))
fake_y = real_y.copy()
fake_y.loc[fake_rn] = 1 - abs(fake_y.loc[fake_rn])

## Part1. Do not consider Noise

def not_consider_noise():

  # Measure the start time
  start_time = time.time()

  X_train, X_test, y_train, y_test = train_test_split(real_X, fake_y , test_size=0.25, random_state=0)

  ss_train = StandardScaler()
  X_train = ss_train.fit_transform(X_train)

  ss_test = StandardScaler()
  X_test = ss_test.fit_transform(X_test)

  model = RandomForestClassifier()

  # Define hyperparameters to search (example hyperparameters for Random Forest)
  param_grid = {'n_estimators': [50, 100],
              'max_depth': [5, 10],
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 2]}

  # Perform grid search for hyperparameter tuning
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
  grid_search.fit(X_train, y_train)

  # Get the best model from grid search
  best_classifier = grid_search.best_estimator_

  # Make predictions on the test set
  y_pred = best_classifier.predict(X_test)

  # Evaluate the model performance
  accuracy = accuracy_score(real_y.loc[y_test.index], y_pred)

  # Measure the end time
  end_time = time.time()

  print("Prediction accuracy for not considering Noise is :", round(accuracy * 100, 3), "%")
  print(f'Total execution time: {end_time - start_time:.4f} seconds')

## Part2. Consider Noise
def consider_noise():

  # Measure the start time
  start_time = time.time()

  split_size, train_size, test_size = 6, 2, 4
  predict_num = int(factorial(split_size - 1) / (factorial(test_size - 1)*factorial(train_size)))
  criteria = 0.8
  split_range = range(0, split_size)

  split = []

  for i in range(0, len(fake_y)):
    split.append(i % split_size)

  random.shuffle(split)

  df_y = pd.DataFrame(fake_y)
  real_X['split'] = split
  df_y['split'] = split

  y_data = df_y
  y_data_prob = df_y.copy()

  split_test = combinations(split_range, test_size)

  for i in split_test:

    ## select
    select = i
    
    ## train
    train_num = []
    train_num = [i for i in split_range if i in select]

    ## test
    test_num = []
    test_num = [i for i in split_range if i not in select]

    print("train : ",  train_num, "test :", test_num)

    ## Data Split
    X_tmp = real_X.copy()
    y_ymp = fake_y.copy()

    X_train = X_tmp[X_tmp['split'].isin(train_num)].drop(columns = 'split')
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    X_test = X_tmp[X_tmp['split'].isin(test_num)].drop(columns = 'split')
    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)
    
    y_train = df_y['target'][df_y['split'].isin(train_num)]
    y_test = df_y['target'][df_y['split'].isin(test_num)]

    ## Model

    model = RandomForestClassifier()
    fit = model.fit(X_train, y_train)
    predictions = fit.predict(X_test)

    # Define hyperparameters to search (example hyperparameters for Random Forest)
    param_grid = {'n_estimators': [50, 100],
                  'max_depth': [5, 10],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 2]}

    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_classifier = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_classifier.predict(X_test)
    y_pred_prob = best_classifier.predict_proba(X_test)

    ## Save Results
    y_test = pd.DataFrame(y_test)
    y_index = y_test.index

    y_predict = pd.DataFrame(y_pred, index = y_index)
    y_predict_prob = pd.DataFrame(y_pred_prob, index = y_index)

    y_data = pd.concat([y_data, y_predict], axis = 1)
    y_data_prob = pd.concat([y_data_prob, y_predict_prob], axis = 1)

  y_data_prob_cat = (y_data_prob.apply(lambda x: x.dropna().reset_index(drop=True), axis = 1)
                    .rename(columns = dict(enumerate(y_data_prob.columns))))
  new_df = y_data_prob_cat.drop(columns = ['target', 'split'])
  cnt = pd.DataFrame(new_df.apply(lambda x: (x > 0.5).sum(), axis = 1))
  final_yn = cnt[0].apply(lambda x : 1 if x >= 4 else 0)

  # Evaluate the model performance
  accuracy = accuracy_score(real_y.loc[y_test.index], final_yn)

  # Measure the end time
  end_time = time.time()

  print("Prediction accuracy for considering Noise is :", round(accuracy * 100, 3), "%")
  print(f'Total execution time: {end_time - start_time:.4f} seconds')

## Final Result
not_consider_noise()
consider_noise()

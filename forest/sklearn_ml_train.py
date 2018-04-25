# IMPORTS

# libraries
import numpy as np
import pandas as pd
# sampling helper function
from sklearn.model_selection import train_test_split
# preprocessing module
from sklearn import preprocessing
# model family - random forest
from sklearn.ensemble import RandomForestRegressor
# cross validation tools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# evaluation metrics (model performance eval)
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
# module to save models for future use
from sklearn.externals import joblib

# LOAD DATA

# data import using panda IO tool for csv
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# pandas DataFrame
data = pd.read_csv(dataset_url, sep=';')

# SPLIT DATA

# target features
y = data.quality
# input features
X = data.drop('quality', axis=1)

# still a dataframe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y)

# DATA PREPROCESSING

# # standardization

# # scaler instance with saved mean, std
# # utility class that implements the Transformer API
# scaler = preprocessing.StandardScaler().fit(X_train)
# # StandardScaler(copy=True, with_mean=True, with_std=True)
# X_train_scaled = scaler.transform(X_train)
# # numpy ndArray
# # [[ 0.51358886  2.19680282 -0.164433   ...  1.08415147 -0.69866131 -0.58608178] ... [-1.73698885 -0.31792985 -0.82867679 ...  1.46964764  1.2491516 2.97009781]]

# X_test_scaled = scaler.transform(X_test)

# PIPELINE
# pipeline handles standardization, and some hyperparameters
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# HYPERPARAMETERS

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}

# CROSS VALIDATION

# set up k folds
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print r2_score(y_test, y_pred)

print mean_squared_error(y_test, y_pred)

# print f1_score(y_test, y_pred)

# print accuracy_score(y_test, y_pred)

joblib.dump(clf, 'rf_regressor.pkl')
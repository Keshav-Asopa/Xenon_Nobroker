#Loading the data
import pandas as pd
k=pd.read_csv('hackathon_rentomter_nobroker.csv')
j=pd.read_csv('finaldataset.csv')
k.columns
s=k['rent']
k=k.drop(['rent'],axis=1)
final=pd.concat([k,j],axis=1)
fullfinal=pd.concat([final,s],axis=1)
fullfinal.to_csv('fullfinal.csv',sep='\t')

#Loading libraries 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

fullfinal.drop(['id','locality','activation_date','amenities'],axis = 1, inplace = True)
X = fullfinal.iloc[:, :-1].values

#getting the last column to predict
y = fullfinal.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

labelencoder_X_3 = LabelEncoder()
X[:, 8] = labelencoder_X_3.fit_transform(X[:, 8])

labelencoder_X_4 = LabelEncoder()
X[:, 9] = labelencoder_X_4.fit_transform(X[:, 9])

labelencoder_X_5 = LabelEncoder()
X[:, 13] = labelencoder_X_5.fit_transform(X[:, 13])

labelencoder_X_6 = LabelEncoder()
X[:, 17] = labelencoder_X_6.fit_transform(X[:, 17])

labelencoder_X_7 = LabelEncoder()
X[:, 18] = labelencoder_X_7.fit_transform(X[:, 18])

#one hot encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [13])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [17])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [18])
X = onehotencoder.fit_transform(X).toarray()

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#best model
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)

import pickle
# save model to file
pickle.dump(model, open("model.pkl", "wb"))
#data processing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 


#importing datasets
dataset=pd.read_csv('Data.csv')

#selecting columns except the last one forming independent variables
X=dataset.iloc[:, :-1].values

#selecting the last column forming dependent variables
y=dataset.iloc[:, 3].values

#filling the missing data

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
 
#encoding the categorical data
#country and purchase column in this case

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
onehotencoder=OneHotEncoder (categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting the dataset into training data and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#scaling of the data set to similar scales and ranges

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



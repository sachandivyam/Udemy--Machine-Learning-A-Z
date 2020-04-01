
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 10)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#fitting linear regression to the training sets

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the salaries for the test set

y_pred=regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.scatter(X_test,y_test,color='black')
plt.scatter(X_test,y_pred,color='yellow')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
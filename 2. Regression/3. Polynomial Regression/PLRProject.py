import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values #make sure x is matrix
y = dataset.iloc[:, 2].values  #y is vector

#check the kind of relationship so obtained
plt.plot(x,y) 

#do not divide into train and test set, because less data and more accuracy is necessary

#make linear regression model just as a reference
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x,y)

#make polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polyreg= PolynomialFeatures(degree=4)
x_poly=polyreg.fit_transform(x)
linreg2=LinearRegression()
linreg2.fit(x_poly,y)

#visualising linear model
plt.scatter(x,y,color='red')
plt.plot(x,linreg.predict(x),color='blue')
plt.show()

#visualising polynomialmodel

plt.scatter(x,y,color='red')
#do not use x_poly as y here because it's already defined and we want to generalise the model for all inputs
plt.plot(x,linreg2.predict(polyreg.fit_transform(x)),color='blue')
plt.show()

linreg.predict(6.5)
y2=linreg2.predict(polyreg.fit_transform(6.5))
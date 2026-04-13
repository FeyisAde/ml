''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets


#how many sameples and How many features?
diabetes = datasets.load_diabetes()
print(diabetes.data.shape)


# What does feature s6 represent?

print(diabetes.DESCR)

# - s6      glu, blood sugar level

#print out the coefficient

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)    

mymodel = LinearRegression()

mymodel.fit(X=X_train, y=y_train)

print(mymodel.coef_)

#print out the intercept

print(mymodel.intercept_)


# create a scatterplot with regression line

predicted = mymodel.predict(X=X_test)

expected = y_test

plt.plot(expected, predicted, ".")

x=np.linspace(0, 330, 100)
y = x
plt.plot(x,y)

plt.show()

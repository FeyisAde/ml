import pandas as pd
import train_test_split from sklearn.model_selection

nyc = pd.read_csv("nyc.csv")

print(nyc.head(3))

print(nyc.Date.values)

print(nyc.Date.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.AverageTemperature.values, random_state=11)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X=X_train, y=y_train)
print(lr.coef_)
print(lr.intercept_)

predicted = lr.predict(X=X_test)
expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")


 # y = mx + b

predict = lambda x: lr.coef_ * x + lr.intercept_
print(predict(2000))
print(predict(1890))

import seaborn as sns

axes = sns.scatterplot(
    data=nyc, 
    x="Date", 
    y="Temperature",
    hue = "Temperature",
    palette = "winter",
    legend = False
)

axes.set_ylim(10, 70)

import numpy as np
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)
plt.show()
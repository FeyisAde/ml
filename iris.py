# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.

#EXERCISE
# load the iris dataset and use classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)

print(iris.data[:3])

print(iris.target[:3])
# to see if the expected and predicted species
# match up
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=11)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.fit(X_train, y_train)

predicted = knn.predict(X_test)

expected = y_test

print(predicted[:20])
print(expected[:20])
# display the shape of the data, target and target_names
print(iris.target_names)
predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]

print(predicted[:20])
print(expected[:20])
# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

# display the values that the model got wrong
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)
# visualize the data using the confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(expected, predicted)
print(confusion)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

confusion_df = pd.DataFrame(confusion, index=iris.target_names, columns=iris.target_names)

figure = plt.figure()

axes = sns.heatmap(confusion_df, annot=True, cmap=plt.cm.nipy_spectral_r)
plt.xlabel("Expected")
plt.ylabel("Predicted")

plt.show()
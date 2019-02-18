from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plot
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Loading

iris = datasets.load_iris()

# Fitting Naive Bayes model to the data

model = GaussianNB()
model.fit(iris.data, iris.target)


expected = iris.target
predicted = model.predict(iris.data)

# Plot

plot.plot(expected, predicted)
print(expected)
print(predicted)
plot.show()

# Cross Validation

print(metrics.classification_report(expected, predicted))

# find the accuracy of classification

print(metrics.confusion_matrix(expected, predicted))
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

# Training the Model

model.fit(X_train, Y_train)

# Training the Model on Testing Set

Y_predicted = model.predict(X_test)

# Evaluating

print("Gaussian Model Accuracy is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)
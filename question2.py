from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
# loading the datasets
data = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20)

# fitting the svm model
svclassifier = svm.SVC(kernel='linear')

# training the model
svclassifier.fit(X_train, y_train)
predict = svclassifier.predict(data.data)

# report the recall
recall_str = classification_report(data.target, predict)
print(recall_str)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


iris = pd.read_csv("iris.data.txt", header=None)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalization.
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Implement SVM algorithm.
clf = svm.SVC()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_predict))

# Task 2.2
pca = PCA(n_components=3)
x = pca.fit_transform(x)
x = pd.DataFrame(x)

# Split and extract the principle component.
x_fpc = x.iloc[:, 0]
x_spc = x.iloc[:, 1]
x_tpc = x.iloc[:, 2]

print(x_fpc)
print(x_spc)
print(x_tpc)


# Task 2.3
x_train, x_test, y_train, y_test = train_test_split(x_fpc, y, test_size=0.2)
x_train = np.array(x_train).reshape((-1, 1))
x_test = np.array(x_test).reshape((-1, 1))
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("First principle accuracy: ", accuracy_score(y_test, y_predict))

x_train, x_test, y_train, y_test = train_test_split(x_spc, y, test_size=0.2)
x_train = np.array(x_train).reshape((-1, 1))
x_test = np.array(x_test).reshape((-1, 1))
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Second principle accuracy: ", accuracy_score(y_test, y_predict))

x_train, x_test, y_train, y_test = train_test_split(x_tpc, y, test_size=0.2)
x_train = np.array(x_train).reshape((-1, 1))
x_test = np.array(x_test).reshape((-1, 1))
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Third principle accuracy: ", accuracy_score(y_test, y_predict))


# Task 2.4
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Three principle components combined accuracy: ", accuracy_score(y_test, y_predict))



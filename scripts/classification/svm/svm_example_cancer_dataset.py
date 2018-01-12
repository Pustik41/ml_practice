import numpy as np
from sklearn import cross_validation, svm
import pandas as pd


df = pd.read_csv('C:/Users/oleksandr.pustovyi/Documents/Udacity/ml_practice/resources/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))      #features
y = np.array(df['class'])             #lable

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)       #shuffle data, and split between test and train data

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [8,2,1,3,2,2,3,2,1]])        #uniq not in dataset
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
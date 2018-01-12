import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X= np.array([[1, 2],
             [1.5, 1.8],
             [5, 8],
             [8, 8],
             [1, 0.6],
             [9, 11]])

#plt.scatter(X[:,0], X[:,1], s=50, linewidths=5)
#plt.show()

clf = KMeans(n_clusters=4)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_


test = np.array([[5,4],
                 [6, 1],
                 [7,4]])

predict = clf.predict(test)


colors = ["g.", "r.", "c.", "b.", "k.", "o."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=150)
for v in range(len(test)):
    plt.plot(test[v][0], test[v][1], colors[predict[v]], markersize=30)
plt.show()


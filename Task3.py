import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = pd.read_csv("iris.data.txt", header=None)
x = np.array(iris.iloc[:, 0:4])
y = np.array(iris.iloc[:, 4])

for i in range(1, 7):
    print("\n<--- K means with", i, "clusters --->")
    k_means = KMeans(n_clusters=i)
    k_means.fit(x)
    # Final centroids
    print("Final centroids:\n", k_means.cluster_centers_)
    # data point labels
    print("Data point labels:\n", k_means.labels_)

print("\n----------------------------------------------\n"
      "Use PCA to reduce the dimension of features and "
      "combine the first, second and "
      "third principal components to implement k-means algorithm:\n")

pca = PCA(n_components=3)
x = pca.fit_transform(x)

for i in range(1, 7):
    print("\n<--- K means with", i, "clusters --->")
    k_means = KMeans(n_clusters=i)
    k_means.fit(x)
    # Final centroids
    print("Final centroids:\n", k_means.cluster_centers_, "\n")
    # data point labels
    print("Data point labels:\n", k_means.labels_, "\n")

    # Draw the scatter plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=k_means.labels_, cmap='rainbow')

    ax.set_xlabel('first principle component')
    ax.set_ylabel('second principle component')
    ax.set_zlabel('third principle component')
    ax.view_init(elev=10, azim=235)
    plt.show()


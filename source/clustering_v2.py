# convolutional autoEncoder를 이용한 clustering

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D     # 3d plot을 위한 package
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from autoEncoder_v4 import *

cvAE = convAutoEncoder()
cvAE.showSummary()
cvAE.train()
cvAE.showComprassPrecision(withReducedData=1)

reducedWindow = cvAE.getReducedData()

# Silhouette method value
S_score = []
# Elbow method value
E_score = []

for i in range(2, 8):
    kmeans_model = KMeans(init='random', n_clusters=i, n_init=10)
    kmeans_model.fit(reducedWindow)
    S_score.append(metrics.silhouette_score(reducedWindow,kmeans_model.labels_,
                                   metric='l1'))
    E_score.append(kmeans_model.inertia_)
sampleClusterNum = 6
kmeans_model = KMeans(init='random', n_clusters=sampleClusterNum, n_init=10)
kmeans_model.fit(reducedWindow)
y_kmeans = kmeans_model.predict(reducedWindow)

fig = plt.figure(figsize=(8, 8))

grid = plt.GridSpec(2,2)
colors = cm.nipy_spectral(y_kmeans.astype(float)/ sampleClusterNum)

# 2D 용
ax = fig.add_subplot(grid[0,0:])
ax.scatter(reducedWindow[:, 0], reducedWindow[:, 1], c=colors)
plt.title(str(cvAE.total_window) + " Windows in " + str(cvAE.reducedDimension)+ " dimension")

# Labeling the clusters (for 2D)
centers = kmeans_model.cluster_centers_
# Draw white circles at cluster centers
ax.scatter(centers[:, 0], centers[:, 1], marker='o',
        c="white", alpha=1, s=100, edgecolor='k')

for i, c in enumerate(centers):
    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=25, edgecolor='k')

# 3D 용 (Cluster labeling 은 제외함)
# ax = fig.add_subplot(grid[0,0:], projection='3d')
# ax.scatter(reducedWindow[:, 0], reducedWindow[:, 1], reducedWindow[: ,2], c=colors )
# plt.title(str(cvAE.total_window) + " Windows in " + str(cvAE.reducedDimension)+ " dimension")

# The vertical line for average silhouette score of all the values

# Compute the silhouette scores for each sample
sample_silhouette_values = metrics.silhouette_samples(reducedWindow,kmeans_model.labels_)

ax2 = fig.add_subplot(grid[1,0])
plt.plot([x for x in range(2,8)], E_score)
plt.ylabel("elbow_scores")
plt.xlabel("Number of centroids")
plt.title("within cluster sum of errors")

# ax1 = fig.add_subplot(grid[0,1])
#
# y_lower = 10
# for i in range(sampleClusterNum):
#     # Aggregate the silhouette scores for samples belonging to
#     # cluster i, and sort them
#     ith_cluster_silhouette_values = \
#         sample_silhouette_values[kmeans_model.labels_ == i]
#
#     ith_cluster_silhouette_values.sort()
#
#     size_cluster_i = ith_cluster_silhouette_values.shape[0]
#     y_upper = y_lower + size_cluster_i
#
#     color = cm.nipy_spectral(float(i) / sampleClusterNum)
#     ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                       0, ith_cluster_silhouette_values,
#                       facecolor=color, edgecolor=color, alpha=0.7)
#
#     # Label the silhouette plots with their cluster numbers at the middle
#     ax1.text(0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#     # Compute the new y_lower for next plot
#     y_lower = y_upper + 10  # 10 for the 0 samples
#
# ax1.axvline(x=S_score[sampleClusterNum-1], color="red", linestyle="--")
# ax1.set_title(str(sampleClusterNum) + "-Means # "+"The silhouette plot")
# ax1.set_xlabel("The silhouette coefficient values")
# ax1.set_ylabel("Cluster label")

plt.subplot(grid[1,1:])
plt.plot([x for x in range(2,8)], S_score)
plt.ylabel("silhouette_score")
plt.xlabel("Number of centroids")
plt.title("clustering performance")
plt.show()

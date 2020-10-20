# 일반적인 Neural Network Autoencoder 를 이용해 K-means clustering 한것.

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # 3d plot을 위한 package
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from autoEncoder_v3 import *

_withLabel = True
nnAE = nnAutoEncoder(trainFile='InsectWingbeatSound_TRAIN.csv',
                     testFile='InsectWingbeatSound_TRAIN.csv',
                     withLabel=_withLabel, reducedDim=32)
nnAE.showSummary()
nnAE.train(epochs=130, batchSize=64)
nnAE.showReducedData()

reducedWindow = nnAE.getReducedData()

# Silhouette method value
S_score = []
# Elbow method value
E_score = []

numOfCluster = 60

for i in range(2, numOfCluster):
    kmeans_model = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans_model.fit(reducedWindow)
    S_score.append(metrics.silhouette_score(nnAE.normalized_X_test, kmeans_model.labels_,
                                            metric='euclidean'))

print(pd.DataFrame(S_score, index=[x for x in range(2, numOfCluster)]))
print('max index {} and max value {}'.format(np.argmax(S_score) + 2, np.max(S_score)))

sampleClusterNum = 39
kmeans_model = KMeans(init='k-means++', n_clusters=sampleClusterNum, n_init=10)
y_kmeans = kmeans_model.fit_predict(reducedWindow)

if _withLabel:
    print('metrics.adjusted_rand_score: {}'.format(metrics.adjusted_rand_score(
        nnAE.true_labels_, y_kmeans
    )))

fig_window = plt.figure(figsize=(sampleClusterNum, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for i, v in enumerate(range(sampleClusterNum)):
    ax1 = fig_window.add_subplot(sampleClusterNum, 2, v + 1)
    for xx in nnAE.normalized_X_test[y_kmeans == v]:
        ax1.plot(xx.ravel(), "k-", alpha=.2)
    ax1.plot(np.mean(nnAE.normalized_X_test[y_kmeans == v], axis=0).ravel(), "r-")
    plt.xlim(0, nnAE.windowSize)
    plt.ylim(-4, 4)
    if i == 0:
        ax1.set_title("Euclidean $k$-means")
plt.show()

fig = plt.figure(figsize=(8, 8))

grid = plt.GridSpec(2, 2)
colors = cm.nipy_spectral(y_kmeans.astype(float) / sampleClusterNum)

# # 2D 용
# ax = fig.add_subplot(grid[0,0:])
# ax.scatter(reducedWindow[:, 0], reducedWindow[:, 1], c=colors )
# plt.title(str(nnAE.total_window) + " Windows in " + str(nnAE.reducedDimension)+ " dimension")
#
# # Labeling the clusters (for 2D)
# centers = kmeans_model.cluster_centers_
# # Draw white circles at cluster centers
# ax.scatter(centers[:, 0], centers[:, 1], marker='o',
#         c="white", alpha=1, s=100, edgecolor='k')
#
# for i, c in enumerate(centers):
#     ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                 s=25, edgecolor='k')

# 3D 용 (3D는 cluster labeling 제외했음.)
ax = fig.add_subplot(grid[0, 0:], projection='3d')
ax.scatter(reducedWindow[:, 0], reducedWindow[:, 1], reducedWindow[:, 2], c=colors)
plt.title(str(nnAE.total_window) + " Windows in " + str(nnAE.reducedDimension) + " dimension")

# The vertical line for average silhouette score of all the values

# Compute the silhouette scores for each sample
sample_silhouette_values = \
    metrics.silhouette_samples(nnAE.normalized_X_test, kmeans_model.labels_
                               , metric='l1')

ax1 = fig.add_subplot(grid[1, 0])

y_lower = 30
for i in range(sampleClusterNum):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[kmeans_model.labels_ == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / sampleClusterNum)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.axvline(x=S_score[sampleClusterNum - 1], color="red", linestyle="--")
ax1.set_title(str(sampleClusterNum) + "-Means # " + "The silhouette plot")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

plt.subplot(grid[1, 1:])
plt.plot([x for x in range(2, numOfCluster)], S_score)
plt.ylabel("silhouette_score")
plt.xlabel("Number of centroids")
plt.title("clustering performance")
plt.show()

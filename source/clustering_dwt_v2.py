# DWT를 이용한 clustering
import time
import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series
from sklearn import metrics
from DWT_v2 import DWT

start_time = time.time()

sample = DWT(num_window='all', window_size=140, dataFile='ECG5000_TEST.csv',
             wvType='haar')

# 9 : 2, 8 : 4, 7 : 8, 6 : 16
sample.reduce_data(object_dimension=25,
                   addition_info=False, lvl=2)
# sample.showComprassPrecision(withReducedData= 1)

# n_clusters : The number of clusters to form as well as the number of centroids to generate.
# n_init : Number of time the k-means algorithm will be run with different centroid seeds.
# The final results will be the best output of n_init consecutive runs in terms of inertia.

# Silhouette method value
S_score = []
# Elbow method value
E_score = []

'''
'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 
'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 
 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski
'''

numOfCluster = 30
seed = 0
np.random.seed(seed)

for i in range(2, numOfCluster):
    print("Euclidean k-means")
    km = TimeSeriesKMeans(n_clusters=i, verbose=True, random_state=0)
    labels_ = km.fit_predict(to_time_series(sample.reducedWindow))

    # kmeans_model = TimeSeriesKMeans(n_clusters=i, metric="softdtw",
    #                        verbose=True, random_state=seed)
    # labels_ = kmeans_model.fit_predict(to_time_series(sample.reducedWindow))
    # S_score.append(silhouette_score(sample.normalizedData, labels_, metric = "softdtw"))

    # print("DBA k-means")
    # dba_km = TimeSeriesKMeans(n_clusters=i, n_init=2, metric="dtw", verbose=True,
    #                           random_state=seed)
    # labels_ = dba_km.fit_predict(sample.reducedWindow)
    # S_score.append(silhouette_score(sample.normalizedData, labels_, metric="dtw"))

    S_score.append(metrics.silhouette_score(sample.normalizedData, labels_, metric='l1'))
    E_score.append(metrics.calinski_harabaz_score(sample.normalizedData, labels_))
    print('S_score : {}'.format(S_score))
print(E_score)
# sampleClusterNum = 30
# kmeans_model = KMeans(init='k-means++', n_clusters=sampleClusterNum, n_init=10)
# kmeans_model.fit(sample.reducedWindow)
# y_kmeans = kmeans_model.predict(sample.reducedWindow)

fig = plt.figure(figsize=(8, 8))

grid = plt.GridSpec(2, 2)
# colors = cm.nipy_spectral(y_kmeans.astype(float)/ sampleClusterNum)
#
# # np.set_printoptions(suppress=True)
# # print(np.array(sample.recoveredWindow[0]))
# # print(sample.windowData[0])
#
# #####
# # 2D 용
# #####
#
# ax = fig.add_subplot(grid[0,0])
# ax.scatter(sample.reducedWindow[:, 0], sample.reducedWindow[:, 1], c=colors)
# plt.title(str(sample.totalWindow) + " Windows in " + str(sample.num_elements)+ " dimension")
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
#
# #########
# # 3D 용
# ##########
#
# # ax = fig.add_subplot(grid[0,0:], projection='3d')
# # ax.scatter(sample.reducedWindow[:, 0], sample.reducedWindow[:, 1], sample.reducedWindow[: ,2], c=colors )
# # plt.title(str(sample.totalWindow) + " Windows in " + str(sample.num_elements)+ " dimension")
#
# #########
# # The vertical line for average silhouette score of all the values
#
# # Compute the silhouette scores for each sample
# sample_silhouette_values = \
#     metrics.silhouette_samples(sample.normalizedData, kmeans_model.labels_,
#                                metric='euclidean')
#
# # labeling 결과 작성
# # with open('Result.csv', 'w') as rs :
# #     for a, b in zip(sample.windowData, kmeans_model.labels_):
# #         rs.write(str(b) + ',' + (','.join(map(str,a))) + '\n')
#
#
# ax1 = fig.add_subplot(grid[1,0])
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

plt.subplot(grid[1, 0])
plt.plot([x for x in range(2, numOfCluster)], E_score)
plt.ylabel("Calinski-Harabaz Score")
plt.xlabel("Number of centroids")
plt.title("Calinski-Harabaz Index")

plt.subplot(grid[1, 1:])
plt.plot([x for x in range(2, numOfCluster)], S_score)
plt.ylabel("silhouette_score")
plt.xlabel("Number of centroids")
plt.title("clustering performance")
plt.show()

elapsed_time = time.time() - start_time
print('Progress time {}'.format(elapsed_time))

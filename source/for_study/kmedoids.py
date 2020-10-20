# https://github.com/rakeshvar/kmedoids/blob/master/kmedoids.py

import numpy as np

import fastdtw
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

######################### K-Medoids
'''
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
Find the intersection of two arrays.
>>> np.argmin(a)
0
>>> np.argmin(a, axis=0)
array([0, 0, 0])
>>> np.argmin(a, axis=1)
array([0, 0])
'''
class KMedoids:
    def __init__(self, n_clusters= 2, batch_size= 300, max_iter= 100,
                tol = 0.01, metric="dtw"):
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol

    def _dist(self, xa, xb):
        if self.metric== 'euclidean':
            return np.sqrt(np.sum(np.square(xa-xb)), axis = -1)
        elif self.metric == 'dtw':
            value, _  = fastdtw.fastdtw(xa, xb, radius=5, dist=(lambda a,b: np.linalg.norm(a-b)))
            return value
        else:
            return np.sum(np.abs(xa - xb))

    def _assign_nearest(self, ids_of_mediods):
        dists = np.empty((0,self.n_clusters), float)
        for xa in self.X :
            temp = np.array([])
            for j in ids_of_mediods:
                temp = np.append(temp, self._dist(xa,self.X[j]))
            dists = np.vstack((dists, temp))
            print(dists.shape)
        # dists = self._dist(self.X[:,None,:], self.X[None,ids_of_mediods,:])

        z = np.argmin(dists, axis=1)
        print(z)
        return z

    def _find_medoids(self, whole_num, assignments):
        # k 개의 클러스터 만큼의 medoid 를 -1로 설정한다.
        medoid_ids = np.full(self.n_clusters, -1, dtype=int)
        # batch size 만큼 whole_num 에서 고른다.
        # 적절한 subset을 찾을때까지 계속해서 고른다(후보군을 잘 골라야함).
        select_subset = True
        while select_subset:
            select_subset = False
            subset = np.random.choice(whole_num, self.batch_size, replace=False)
            indices = []
            for i in range(self.n_clusters):
                temp = np.intersect1d(np.where(assignments==i)[0], subset)
                if temp.size == 0:
                    select_subset = True
                    break
                indices.append(temp)

        for i, indice in enumerate(indices):
            # id 가 i 인 cluster에 할당된 data가 있는지 확인한다. (* 없을 수도 있음 !)
   #         indicew = np.intersect1d(np.where(assignments==i)[0], subset)

            # cluster i 안에 있는 data들 끼리의 모든 길이(distance)를 다 재본다.
            dist = np.empty((0,len(indice)),float)
            for w in indice :
                temp = np.array([])
                for j in indice:
                    # 이 부분이 많이 늦다.
                    temp = np.append(temp, self._dist(self.X[w] , self.X[j]))
                dist = np.vstack((dist, temp))
                print(dist.shape)
            # 각 data 들이 가지는 다른 data와의 거리를 모두 합한다.
            # 즉, cluster i에 n 개의 data들이 있다면 n 개의 거리 값이 나온다.
            distances = dist.sum(axis=0)
            # distances = self._dist(self.X[indices, None, :], self.X[None, indices, :]).sum(axis=0)

            # 이 거리 중에서 최소를 가지는 index를 찾아내어 그 data를 medoid_id로 정한다.
            medoid_ids[i] = indice[np.argmin(distances)]

        return medoid_ids

    def fit(self, X):
        whole_num = X.shape[0]
        if self.batch_size == 'all':
            self.batch_size = whole_num
        self.X = X

        print("Initializing to random medoids.")
        ids_of_medoids = np.random.choice(whole_num, self.n_clusters, replace=False)
        # 랜덤하게 고른 medoid 들
        print(ids_of_medoids)
        # 이 기준점을 이용해서 근처에 있는 data를 assign 하는 작업을 한다.
        class_assignments = self._assign_nearest(ids_of_medoids)

        # 각각의 data들은 이제 특정 medoid에 assign 되있는 상태 (index 로 표현됨.)
        for i in range(self.max_iter):
            print("\tFinding new medoids.")
            # 새로운 medoid를 찾아본다.
            ids_of_medoids = self._find_medoids(whole_num, class_assignments)
            # 새로찾은 medoid 에게 기존의 assign된 data들을 다시 배분해본다.
            print("\tReassigning points.")
            new_class_assignments = self._assign_nearest(ids_of_medoids)

            # 다름의 정도를 다시 재 측정한다.
            diffs = np.mean(new_class_assignments != class_assignments)
            # 이전 assign 된 내용은 삭제되고 새롭게 assign된 내용이 main이 된다.
            class_assignments = new_class_assignments

            # 만약 이전과 assign 된 내용이 별로 차이가 없다면 clustering 종료.
            print("iteration {:2d}: {:.2%} of points got reassigned."
                  "".format(i, diffs))
            if diffs <= self.tol:
                break

        return class_assignments, ids_of_medoids





if __name__ == '__main__':

    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    print(x)
    y = np.array([[2, 2], [3, 3], [4, 4]])
    distance, path = fastdtw.fastdtw(x, y, dist=euclidean)
    print(distance)

    ######################### Generate Fake Data
    print("Initializing Data.")
    # dimension
    d = 16
    k = 6
    n = k * 1000
    batch_sz = 1000
    x = np.random.normal(size=(n, d))
    EUCLIDEAN = False

    print("n={}\td={}\tk={}\tbatch_size={} ".format(n, d, k, batch_sz))
    print("Distance metric: ", "Eucledian" if EUCLIDEAN else "Manhattan")

    print("\nMaking k-groups as:")
    for kk in range(k):
        dd = (kk-1)%d
        print("    x[{}:{}, {}] += {}".format(kk*n//k, (kk+1)*n//k, dd , 3*d*kk))
        x[kk*n//k:(kk+1)*n//k,dd] += 3*d*kk

    KMedoids_model = KMedoids(n_clusters=k, metric='dtw', batch_size= batch_sz)
    ######################### Fitting
    print("\nFitting Kmedoids.")
    final_assignments, final_medoid_ids = KMedoids_model.fit(x)

    print("\nFitting Kmeans from Scikit-Learn")
    fit = KMeans(n_clusters=k).fit(x)
    kmeans_assignments = fit.labels_
    kmeans = fit.cluster_centers_

    mismatch = np.zeros((k, k))
    for i, m in (zip(final_assignments, kmeans_assignments)):
        mismatch[i, m] += 1

    np.set_printoptions(suppress=True)
    print("\nKMedoids:")
    print(x[final_medoid_ids, ])
    print("K-Medoids class sizes:")
    print(mismatch.sum(axis=-1))
    print("\nKMeans:")
    print(kmeans)
    print("K-Means class sizes:")
    print(mismatch.sum(axis=0))
    print("\nMismatch between assignment to Kmeans and Kmedoids:")
    print(mismatch)
    print("Should ideally be {} * a permutation matrix.".format(n//k))

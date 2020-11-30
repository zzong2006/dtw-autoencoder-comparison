import random
import numpy as np


class CustomKMeans:
    def __init__(self, init='k-means++', n_clusters=2, n_init=10):
        self.init = init
        self.n_clusters = n_clusters

    def fit(self, X):
        # find initial centroids of K-means
        self.X = X
        if self.init == 'k-means++':
            self.mu = random.sample(X, 1)
            while len(self.mu) < self.n_clusters:
                self._dist_from_centers()
        # do K-means clustering
        while not self._has_converged():
            self.oldmu = self.mu
            # Assign all points in X to clusters and reEvaluate center
            self._cluster_points()

    def _choose_next_center(self, dis):
        probs = dis / dis.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind]

    def _dist_from_centers(self):
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in self.mu]) for x in self.X])
        self.mu.append(self._choose_next_center(D2))

    def _has_converged(self):
        K = len(self.oldmu)
        return (set([tuple(a) for a in self.mu]) == set([tuple(a) for a in self.oldmu])
                and len(self.mu) == K)

    def _cluster_points(self):
        mu = self.mu
        clusters = {}
        for x in self.X:
            bestmukey = min([(i, np.linalg.norm(x - mu[i])) for i in range(len(mu))],
                            key=lambda t: t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self._reevaluate_centers(clusters)

    def _reevaluate_centers(self, clusters):
        newmu = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))
        self.mu = newmu

    # Euclidean Distance Caculator
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

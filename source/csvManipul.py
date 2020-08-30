import numpy as np

def eblowScore(n_clusters, X, labels):
    label_num = np.bincount(labels)
    for i in range (n_clusters):
        for j in range(X.shape[0]-1):
            for w in range(j, X.shape[0]):
                dist = np.linalg.norm(X[j] - X[w])
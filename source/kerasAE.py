# Author: Romain Tavenard
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

sampleClusterNum = 11
X_scaled = np.random.randn(1980, 256)
labels_ = np.random.randint(0, high=11, size=1980)
plt.figure(figsize=(sampleClusterNum, sampleClusterNum * 2 - sampleClusterNum / 2))
for i, v in enumerate(range(sampleClusterNum)):
    plt.subplot(sampleClusterNum, 2, v + 1)
    for xx in X_scaled[labels_ == v]:
        # print(xx)
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(np.mean(X_scaled[labels_ == v], axis=0).ravel(), "r-")
    plt.xlim(0, 3)
    plt.ylim(-4, 4)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    if i == 0:
        plt.title("Euclidean $k$-means")

plt.show()

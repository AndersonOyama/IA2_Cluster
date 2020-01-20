import numpy as np
import pandas as pd
import sys
import math
import random

import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from itertools import cycle





def main():
    # redWine = pd.read_csv("../Dataset/winequality-red.csv", ";")
    # whiteWine = pd.read_csv("../Dataset/winequality-white.csv", ";")
    dataset = pd.read_csv("../Dataset/s1.txt", sep="    ", header=None)
 
    # treino, ans = gerador_treino(redWine, whiteWine)

    plt.scatter(dataset.iloc[:,0].values, dataset.iloc[:,1].values, label="True position")
    plt.show()



    bandwidth = estimate_bandwidth(dataset, quantile=0.05, n_samples=2500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(dataset)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)


    plt.figure(1)
    plt.clf()



    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(dataset.iloc[my_members, 0].values, dataset.iloc[my_members, 1].values, col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()




def gerador_treino(redWine, whiteWine):
    baseTreino = []
    baseClasse = []
    ansRed = [0] * len(redWine)
    ansWhite = [1] * len(whiteWine)
    base = pd.concat([redWine, whiteWine], ignore_index=True)
    classe = ansRed + ansWhite
    baseTreino, baseClasse = shuffle(base, classe)
    return baseTreino, baseClasse




if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import sys
import math
import random


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle



import matplotlib.pyplot as plt
import scikitplot as skplt


def main():
    porcentagem = float(sys.argv[1])
    redWine = pd.read_csv("../Dataset/winequality-red.csv", ";")
    whiteWine = pd.read_csv("../Dataset/winequality-white.csv", ";")

    treino, ans = gerador_treino(redWine, whiteWine)

    plt.scatter(treino.iloc[:,0].values, treino.iloc[:,1].values, label="True position")
    plt.show()



    bandwidth = estimate_bandwidth(treino, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(treino)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)




    X_train, X_test, y_train, y_test = train_test_split(treino, ans, test_size=porcentagem, random_state=0)




    # plt.scatter(treino.iloc[:,0].values,treino.iloc[:,1].values, c=labels_unique, cmap='rainbow')
    # plt.show()


    plt.figure(1)
    plt.clf()

    from itertools import cycle

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(treino.iloc[my_members, 0].values, treino.iloc[my_members, 1].values, col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


    print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))




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

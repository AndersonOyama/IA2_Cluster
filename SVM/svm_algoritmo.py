import numpy as np
import pandas as pd
import sys
import math
import random


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


from utils import read_data, plot_data, plot_decision_function

import matplotlib.pyplot as plt
import scikitplot as skplt


def main():
    porcentagem = sys.argv[1]
    redWine = pd.read_csv("../Dataset/winequality-red.csv", ";")
    whiteWine = pd.read_csv("../Dataset/winequality-white.csv", ";")

    treino, ans = gerador_treino(redWine, whiteWine)

    plt.scatter(treino.iloc[:,0].values, treino.iloc[:,1].values, label="True position")
    plt.show()


    bandwidth = estimate_bandwidth(treino, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(treino)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = 2

    print("estimativa de cluster: %d" % n_clusters_)


    X_train, X_test, y_train, y_test = train_test_split(treino, ans, test_size=porcentagem, random_state=0)



    clf = KMeans(n_clusters=2)
    clf.fit(X_train, y_train)
    y_pred = clf.fit_predic(X_test)
    print("Acuracia: {}".format(accuracy_score(y_test, y_pred)))


    # plt.scatter(treino.iloc[:,0].values,treino.iloc[:,1].values, c=kmeans.labels_, cmap='rainbow')
    # plt.show()




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

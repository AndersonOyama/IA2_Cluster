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
    localset = "../Dataset/"
    datasetLocal = localset + sys.argv[1]
    dataset = pd.read_csv(datasetLocal, sep="    ", header=None)
    sample = int(sys.argv[2])
    mediumDistance = float(sys.argv[3])

    plt.scatter(dataset.iloc[:,0].values, dataset.iloc[:,1].values, label="")
    plt.title("Grafico de: " + sys.argv[1])
    plt.show()


    bandwidth = estimate_bandwidth(dataset, quantile=mediumDistance, n_samples=sample)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(dataset)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    plt.figure(1)
    plt.clf()


    colors = []
    for i in range(n_clusters_):
        r = lambda: random.randint(0,255)
        color = '#%02X%02X%02X' % (r(),r(),r())
        if((color not in colors) and (color != '#ffffff')):
            colors.append(color)
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(dataset.iloc[my_members, 0].values, dataset.iloc[my_members, 1].values, col, marker='.', linestyle=None, linewidth=0)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Número estimado de cluster: {}\nQuantidade de amostra: {}\nMédia de distancia: {}'.format(n_clusters_,sample, mediumDistance))
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
    if(len(sys.argv) < 4):
        print("Execução inválida. Exemplo: python3 MeanShift.py nome_do_arquivo_teste quantidade_de_amostra media_de_distancia[0,1]")
        exit(1)
    else:
        main()

import numpy as np
import pandas as pd
import sys
import math
import random

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
import scikitplot as skplt


def main():
    porcentagem = float(sys.argv[1])
    redWine = pd.read_csv("../Dataset/winequality-red.csv", ";")
    whiteWine = pd.read_csv("../Dataset/winequality-white.csv", ";")

    # treinoRed, treinoWhite, testeRed, testeWhite, answerRed, answerWhite,
    treino, ans = gerador_treino(redWine, whiteWine)

    X_train, X_test, y_train, y_test = train_test_split(treino, ans, test_size=porcentagem, random_state=0)

    clf = svm.SVC(kernel='linear', gamma=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))

    X, y = make_blobs(n_samples=( int(porcentagem * len(treino))), centers=2, random_state=0, cluster_std=0.60)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
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

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

from utils import read_data, plot_data, plot_decision_function

import matplotlib.pyplot as plt
import scikitplot as skplt


def main():
    porcentagem = float(sys.argv[1])
    redWine = pd.read_csv("../Dataset/winequality-red.csv", ";")
    whiteWine = pd.read_csv("../Dataset/winequality-white.csv", ";")

    treino, ans = gerador_treino(redWine, whiteWine)

    X_train, X_test, y_train, y_test = train_test_split(treino, ans, test_size=porcentagem, random_state=0)


    plot_data(X_train, y_train, X_test, y_test, porcentagem)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    plot_decision_function(X_train, y_train, X_test, y_test, clf)
    
    y_pred = clf.predict(X_test)



    print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))

    # plot_svc_decision_function(test)



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

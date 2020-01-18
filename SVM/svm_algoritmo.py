import numpy as np
import pandas as pd
import sys
import math
import random

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    porcentagem = float(sys.argv[1])
    entrada1 = pd.read_csv("../Dataset/winequality-red.csv", ";")
    entrada2 = pd.read_csv("../Dataset/winequality-white.csv", ";")
    treinoRed, treinoWhite, testeRed, testeWhite = gerador_treino(entrada1, entrada2, porcentagem)

    X_train, X_test, y_train, y_test = train_test_split(entrada1, entrada1, test_size=0.25, random_state=100)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
    print(y_pred)
    print(X_test)
    print(X_train)



def gerador_treino(wine1, wine2, porcentagem):
    baseRed = []
    baseWhite = []
    testeRed = []
    testWhite = []
    
    sortIndex = randomIndex(len(wine1), math.ceil(len(wine1)*porcentagem))
    for i in range(0, len(wine1)):
        if (i in sortIndex):
            baseRed.append(wine1.loc[(i),:])
        else:
            testeRed.append(wine1.loc[(i),:])
    
    sortIndex = randomIndex(len(wine2), math.ceil(len(wine2)*porcentagem))
    for i in range(0, len(wine2)):
        if (i in sortIndex):
            baseWhite.append(wine2.loc[(i),:])
        else:
            testWhite.append(wine2.loc[(i),:])    

    return (baseRed, baseWhite, testeRed, testWhite)


def randomIndex(size, qtd):
    index = random.sample(range(0,size), qtd)
    index.sort()
    return index
    

def randomBool():
    random_bit = random.getrandbits(1)
    random_boolean = bool(random_bit)
    return(random_boolean)

if __name__ == '__main__':
    main()

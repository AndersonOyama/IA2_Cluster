import numpy as np
import pandas as pd
import sys
import math
import random

from sklearn import svm

def main():

    treino_entrada1 = pd.read_csv("../Dataset/winequality-red.csv", ";")
    treino_entrada2 = pd.read_csv("../Dataset/winequality-white.csv", ";")
    # print(treino_entrada1.loc[1,:])
    treinoRed, treinoWhite = gerador_treino(treino_entrada1, treino_entrada2, float(sys.argv[1]))
    print(treinoRed)
    categorias = ["red","white"]


def gerador_treino(wine1, wine2, porcentagem):
    baseRed = []
    baseWhite = []
    i = 0
    while (len(baseRed) < math.ceil(len(wine1)*porcentagem)):
        if(i > len(wine1)):
            i = 0
        if ((randomBool()) == True):
            baseRed.append(wine1.loc[(i), :])
            i += 1
        i += 1
    i = 0
    while (len(baseWhite) < math.ceil(len(wine2)*porcentagem)):
        if(i>len(wine2)):
            i=0
        if ((randomBool()) == True):
            baseWhite.append(wine2.loc[(i), :])
            i += 1
        i += 1
    return (baseRed, baseWhite)


def randomBool():
    random_bit = random.getrandbits(1)
    random_boolean = bool(random_bit)
    return(random_boolean)

if __name__ == '__main__':
    main()

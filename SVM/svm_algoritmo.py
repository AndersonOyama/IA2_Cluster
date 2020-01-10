import numpy as np
import pandas as pd
import sys

from sklearn import svm
from sklearn.feature_extraction.dict_vectorizer import BaseEstimator
from sklearn.naive_bayes import BernoulliNB


def main():

    treino_entrada = pd.read_csv("../Dataset/wine.csv", ";")
    dados = pd.read_csv("",';')
    categorias = ["1", "2", "3"]
    vetorizador = BaseEstimator(binary = 'true')
    treino = treino_entrada(treino_entrada, vetorizador)

def treinamento(dados_treino, vetorizador):
    treino_vinho = [dados_treino[0] for dados_treino in dados_treino]

    treino_vinho = vetorizador.fit_transform(treino_vinho)
    return BernoulliNB().fit(treino_vinho)

if __name__ == '__main__':
    main()

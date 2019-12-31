import numpy as np
import pandas as pd
import sys
from sklearn import svm


def main():

    dataset = pd.read_csv("Dataset/wine.csv",";")
    print(dataset)
    dataset.shape
    dataset.head()

if __name__ == '__main__':
    main()

import sys
from sklearn import metrics
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if (len(sys.argv) < 2):
		print("É necessário informar um dataset")
		
else:
		dataset = pd.read_csv(sys.argv[1], sep="    ", header=None, engine="python")

		print("Dataset: ***********************************")
		print(dataset)
		print("********************************************\n")
		
		#plt.scatter(dataset.iloc[:,0].values, dataset.iloc[:,1].values, label="")
		#plt.title("Grafico de: " + sys.argv[1])
		#plt.show()

		# n_jobs = -1?
		print("KMeans Centroids: **************************")
		kmeans = KMeans(n_clusters=15, random_state=42).fit(dataset)
		print(kmeans.cluster_centers_)
		print("********************************************")

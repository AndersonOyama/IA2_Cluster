import sys
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def split_sets(dataset):
		dataset = dataset.sample(frac=1)
		train_len = int(len(dataset) * 0.25)
		train_set = dataset.iloc[:train_len]
		test_set = dataset.iloc[train_len:]
		return train_set, test_set

if (len(sys.argv) < 3):
		print("É necessário informar um dataset")
else:
		dataset = pd.read_csv(sys.argv[1], sep="    ", header=None, engine="python")
		train_set, test_set = split_sets(dataset)
		kmeans = KMeans(n_clusters=int(sys.argv[2]), random_state=42).fit(train_set)
		Y = kmeans.predict(test_set)
		n_clusters_ = len(np.unique(kmeans.labels_))

		colors = []
		for i in range(n_clusters_):
				r = lambda: random.randint(0,255)
				color = '#%02X%02X%02X' % (r(),r(),r())
				if((color not in colors) and (color != '#ffffff')):
						colors.append(color)

		for k, col in zip(range(n_clusters_), colors):
				my_members = Y == k
				cluster_center = kmeans.cluster_centers_[k]

				plt.plot(test_set.iloc[my_members, 0].values, test_set.iloc[my_members, 1].values, col,
								 marker='.', linestyle=None, linewidth=0)
				plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
		plt.title(f'Grafico do dataset: {sys.argv[1]}')
		plt.show()

"""
Reference: https://tslearn.readthedocs.io/en/latest/index.html
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
from tslearn.clustering import TimeSeriesKMeans, GlobalAlignmentKernelKMeans, KShape, silhouette_score
from tslearn.datasets import CachedDatasets
from tslearn.metrics import sigma_gak
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax, TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
import code
import datetime


data_path = "./"
out_path = "./"

# data_path = "../data/"
# out_path = "../outputs/"

seed = 0
np.random.seed(seed)

# df = pd.read_hdf("filtered_data.hdf5", key="zeal")
xtrain = pickle.load(open(data_path + "training_data.pck","rb"))
ytrain = pickle.load(open(data_path + "training_labels.pck","rb"))

# x_train = TimeSeriesScalerMinMax().fit_transform(xtrain[:260]) #shapes comparison
x_train = TimeSeriesScalerMeanVariance().fit_transform(xtrain[:500]) #variance comparison
x_train = TimeSeriesResampler(sz=500).fit_transform(x_train)
sz = x_train.shape[1]

print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=10,
                          n_init=1,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)

y_pred = dba_km.fit_predict(x_train)

plt.figure()
for yi in range(10):
    plt.subplot(10, 1, yi+1)
    for xx in x_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")
plt.show()

# distortion = []
# sil_score = []
# for c in range(2,20):
#     print("***********************************")
#     print("No. of clusters = {}".format(c))
#     # dba_km = TimeSeriesKMeans(n_clusters=c,
#     #                       n_init=1,
#     #                       metric="dtw",
#     #                       verbose=True,
#     #                       max_iter_barycenter=100,
#     #                       random_state=seed)

#     # dba_km = GlobalAlignmentKernelKMeans(n_clusters=c,
#     #                                  sigma=sigma_gak(x_train),
#     #                                  n_init=20,
#     #                                  verbose=True,
#     #                                  random_state=seed)

#     dba_km = KShape(n_clusters=c, verbose=True, random_state=seed)

#     y_pred = dba_km.fit_predict(x_train)
#     # code.interact(local=locals())

#     distortion_avg = dba_km.inertia_
#     sil_avg = silhouette_score(x_train, y_pred, metric="dtw")

#     print("Distortion = {}".format(distortion_avg), flush=True)
#     print("Silhouette Score = {}".format(sil_avg), flush=True)

#     distortion.append(dba_km.inertia_)
#     sil_score.append(silhouette_score(x_train, y_pred))

#     # if c%50 ==0:
#     #     pickle.dump(distortion, open(out_path + "distortion_clustering.pck","wb"))
#     #     pickle.dump(sil_score, open(out_path + "silhouette_score_clustering.pck","wb"))
#     #     print("Saved checkpoint")

# pickle.dump(distortion, open(out_path + "distortion_clustering_final.pck","wb"))
# pickle.dump(sil_score, open(out_path + "silhouette_score_clustering_final.pck","wb"))

code.interact(local=locals())

plt.plot(distortion, marker="*")
plt.show()
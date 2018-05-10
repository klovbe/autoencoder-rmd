#encoding=utf-8

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import metrics
import csv
import os


run_type =  "prediction_10.0_0_epoch800_batch256"
save_dir = "F:/project/autoencoder-rmd/result"
# name_list = ["human3","human4","mouse1","mouse2"]
# name_list = ["brain","kolod","pollen","usoskin","zelsel","human1","human2","human3","human4","mouse1","mouse2"]
name_list = ["brain","kolod","pollen","usoskin"]
# drop_list = [0.2,0.4,0.6,0.8]
# mask_list = [0.2,0.4,0.6,0.8]
out_list = ["complete","encoder.out"]
cluster_method = ["tsne", "pca"]

logdir = os.path.join(save_dir,run_type)
if os.path.exists(logdir) is False:
    os.mkdir(logdir)
logfile = open(logdir + '/cluster_result.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['name', 'out', 'method', 'acc', 'nmi', 'ari'])
# logwriter.writeheader()

for name in name_list:
    label_df = pd.read_csv("F:/project/data/{}_label.csv".format(name), index_col=None, header=None)
    label = label_df.values
    label = np.squeeze(label)
    if not isinstance(label, (int, float)):
        label = LabelEncoder().fit_transform(label)
    n_clusters = len(np.unique(label))
    for out in out_list:
        df = pd.read_csv("F:/project/autoencoder-rmd/{}/h_{}/h_{}/h_{}.{}".format(
                run_type, name, name, name, out))
        X = df.values
        for method in cluster_method:
            if method == 'tsne':
                X = PCA(n_components=50).fit_transform(X)
                X_embedded = TSNE(n_components=2).fit_transform(X)
                y_pred =  KMeans(n_clusters=n_clusters, n_init=40).fit_predict(X_embedded)
            else:
                X_embedded = PCA(n_components=n_clusters).fit_transform(X)
                y_pred =  KMeans(n_clusters=n_clusters, n_init=40).fit_predict(X_embedded)
            acc = np.round(metrics.acc(label, y_pred), 5)
            nmi = np.round(metrics.nmi(label, y_pred), 5)
            ari = np.round(metrics.ari(label, y_pred), 5)
            logdict = dict(name=name, out=out, method=method , acc=acc, nmi=nmi, ari=ari)
            logwriter.writerow(logdict)

logfile.close()




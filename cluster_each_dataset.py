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


run_type = "prediction_10.0_0_epoch800_batch256"
save_dir = "F:/project/autoencoder-rmd/result"
# name_list = ["human3","human4","mouse1","mouse2"]
# name_list = ["brain","kolod","pollen","usoskin","zelsel","human1","human2","human3","human4","mouse1","mouse2"]
name_list = ["brain","kolod","pollen","usoskin"]
# drop_list = [0.2,0.4,0.6,0.8]
# mask_list = [0.2,0.4,0.6,0.8]
out_list = ["complete","encoder.out"]
cluster_method = ["tsne", "pca"]

logdir = os.path.join(save_dir,"each_dataset")
if os.path.exists(logdir) is False:
    os.mkdir(logdir)
logfile = open(logdir + '/cluster_result_direct_tsne.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['name', 'gamma', 'out', 'method', 'acc', 'nmi', 'ari'])
logwriter.writeheader()
gamma_list = [0.0, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 8.0, 20.0]

for name in name_list:
    print("dataset : {}".format(name))
    data_df = pd.read_csv("F:/project/Gan/data/h_" + name + ".train")
    data = data_df.values
    samples, feature_nums = data.shape
    print("dataset consists of {} samples, {} genes".format(samples, feature_nums))
    zero_rate = np.sum(data == 0)/samples/feature_nums
    print("zero rate : {}".format(zero_rate))
    label_df = pd.read_csv("F:/project/data/{}_label.csv".format(name), index_col=None, header=None)
    label = label_df.values
    label = np.squeeze(label)
    if not isinstance(label, (int, float)):
        label = LabelEncoder().fit_transform(label)
    n_clusters = len(np.unique(label))
    print("{} clusters in total".format(n_clusters))
    for gamma in gamma_list:
        for out in out_list:

            df = pd.read_csv("F:/project/autoencoder-rmd/each_dataset/h_{}/{}/h_{}/h_{}.{}".format(
                    name, gamma, name, name, out))
            X = df.values
            if out == 'complete':
                zero_rate_impute = np.sum(X == 0) / samples / feature_nums
                print("zero rate of imputed data, gamma = {} : {}".format(gamma, zero_rate_impute))

            for method in cluster_method:
                if method == 'tsne':
                    # X = PCA(n_components=50).fit_transform(X)
                    X_embedded = TSNE(n_components=2).fit_transform(X)
                    y_pred =  KMeans(n_clusters=n_clusters, n_init=40).fit_predict(X_embedded)
                else:
                    X_embedded = PCA(n_components=n_clusters).fit_transform(X)
                    y_pred =  KMeans(n_clusters=n_clusters, n_init=40).fit_predict(X_embedded)
                acc = np.round(metrics.acc(label, y_pred), 5)
                nmi = np.round(metrics.nmi(label, y_pred), 5)
                ari = np.round(metrics.ari(label, y_pred), 5)
                logdict = dict(name=name, gamma=gamma, out=out, method=method , acc=acc, nmi=nmi, ari=ari)
                logwriter.writerow(logdict)

logfile.close()




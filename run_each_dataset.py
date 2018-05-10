#encoding=utf-8

import os
import codecs
import subprocess
from time import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import metrics
import csv


data_dir = "F:/project/Gan/data"
python_path = "C:/ProgramData/Anaconda/python.exe"
script_path = "F:/project/autoencoder-rmd/train.py"
done_path = "F:/project/autoencoder-rmd/done_train.txt"
done_model_names = []

# name_list = os.listdir(data_dir)
gamma_list = [0.0, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 8.0, 20.0]
name_list = ["h_pollen.train", "h_kolod.train", "h_brain.train", "h_usoskin.train"]
# name_list = ["h_kolod.train"]


for file in name_list:
    path = os.path.join(data_dir, file)
    name = file.split(".")[0]
    if os.path.exists("./run_logs/each_dataset/{}".format(name)) is False:
        os.mkdir("./run_logs/each_dataset/{}".format(name))
    for gamma in gamma_list:
        outDir = "./each_dataset/{}/{}".format(name,gamma)
        if name in done_model_names:
          print("{} has been trained before".format(name))
          continue
        par_dict = {
        "python": python_path,
        "script_path": script_path,
        "model_name": name,
        "datapath":path,
        "outDir": outDir,
        "gamma":gamma
        }
        cmd = "python {script_path} --model_name={model_name}  --train_datapath={datapath} " \
            " --outDir={outDir} --epoch=1000 --batch_size=128 --gamma={gamma} > ./run_logs/each_dataset/{model_name}/{gamma}.log 2>&1".format(
        **par_dict
            )
        print("running {}...".format(cmd))
        ret = subprocess.check_call(cmd, shell=True, cwd="F:/project/autoencoder-rmd")
        print(ret)



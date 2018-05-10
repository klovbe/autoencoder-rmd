
# coding: utf-8

import os
import sys
import plot

import pandas as pd
import numpy as np
import codecs
import tensorflow as tf

import utils
import argparse
from scipy.stats import norm
from Dataset import DataSet
from autoencoder import AutoEncoder
#
# model_class_dict = {
#   "Gan": Gan,
#   "autoencoder": AutoEncoder
# }

flags = tf.app.flags
flags.DEFINE_integer("epoch", 800, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_string("model_class", "autoencoder", "model class[autoencoder|gan]")
flags.DEFINE_integer("batch_size", 256, "The size of batch data [64]")
flags.DEFINE_integer("feature_nums", None, "The dimension of data to use")
flags.DEFINE_string("activation", "relu", "auto encoder activation")
flags.DEFINE_string("train_datapath", "data/drop80-0-1.train", "Dataset directory.")
# flags.DEFINE_string("infer_complete_datapath", "data/drop80-0-1.infer", "path of infer complete path")
# flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_name", "auto_encoder0", "model name will make dir on checkpoint_dir")

# flags.DEFINE_float("truly_mis_pro", -1.0 , "the prob of truly missing values for value 0 [ truly_mis_pro <=0 mean don't random mask, all is missing]")
# flags.DEFINE_string("random_mask_path", "None", "the path of probs of letting value 0's prob trained[None]")
flags.DEFINE_float("dropout", 2.0, "dropout layer prob > 1 mean no layer")
# flags.DEFINE_float("random_sample_mu",0.0,"random mu")
# flags.DEFINE_float("random_sample_sigma", 1.0, "random sigma")
# flags.DEFINE_float("normal_factor", 1e6, "normal factor")

flags.DEFINE_boolean("plot_complete", False, "plot fig after every test or prediction")
# flags.DEFINE_integer("test_freq_steps", 500, "test freq steps")
flags.DEFINE_integer("save_freq_steps", 100, "save freq steps")
flags.DEFINE_integer("log_freq_steps", 10, "log freq steps")

flags.DEFINE_boolean("load_checkpoint", False, "if have checkpoint, whether load prev model first")
# flags.DEFINE_integer("sample_steps", 500, "every sample_steps, will sample the generate mini batch data")
flags.DEFINE_float("gpu_ratio", 1.0, "per_process_gpu_memory_fraction[1.0]")
flags.DEFINE_float("gamma",10.0, "tuning parameter of sparse_loss")
flags.DEFINE_float("beta", 0.0, "tuning parameter of rank_loss")

flags.DEFINE_string('outDir', 'prediction', "output dir")

FLAGS = flags.FLAGS

# model_class = model_class_dict[FLAGS.model_class]
print("sampling dataset...")
sample_dataset = DataSet(FLAGS.train_datapath, nrows=64)
feature_nums = FLAGS.feature_nums or sample_dataset.feature_nums
print(np.linalg.matrix_rank(sample_dataset.data))


model = AutoEncoder(feature_nums, gamma=FLAGS.gamma, beta=FLAGS.beta, dropout= FLAGS.dropout,learning_rate=FLAGS.learning_rate,
                    activation=FLAGS.activation, model_name=FLAGS.model_name)
model.train(FLAGS)


#encoding=utf-8

import os
import sys
import plot
import data_preprocess

import pandas as pd
import numpy as np
import tensorflow as tf
from Dataset import DataSet
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import constraints
from keras import backend as K
from tensorflow.contrib import distributions

import utils
from layers import *
import shutil
import time
from datetime import datetime

activation_dict = {
"tanh":tf.tanh,
"sigmoid": tf.sigmoid,
"relu": tf.nn.relu
}

class AutoEncoder(object):

  def __init__(self, feature_num, gamma=0.01, beta=0.01, dropout = None,
               learning_rate=0.001, activation="relu", model_name="auto_encoder", **kwargs):

    self.feature_num = feature_num
    self.gamma = gamma
    self.beta = beta
    self.activation = activation_dict[activation]
    self.learning_rate = learning_rate
    self.model_name = model_name
    self.create_conf = kwargs
    self.dropout = dropout
    self._create_model()

  def _create_model(self):
    # tf Graph input (only pictures)
    self.is_training = tf.placeholder(tf.bool, name="is_training")
    self.X = tf.placeholder(tf.float32, [None, self.feature_num], name="X")
    self.mask = tf.placeholder(tf.bool, [None, self.feature_num], name="mask")
    # self.keep_bools = tf.placeholder(tf.float32, [None, self.feature_num], name="keep_bools")

    # self.encoder_out = self.encoder(self.X)  # through activation
    # self.decoder_out = self.decoder(self.encoder_out)  # must not through activation


    self.encoder_out = self.encoder_bn(self.X) #through activation
    self.decoder_out = self.decoder_bn(self.encoder_out)  #must not through activation

    # origin_nums = tf.reduce_sum(self.mask)

    # t_mask = tf.logical_or( tf.cast(self.mask, tf.bool) , tf.cast(self.keep_bools, tf.bool) )
    # t_mask = tf.cast(t_mask, tf.float32)

    # if mis_pro is not None:
    #   bern = distributions.Bernoulli(1 - mis_pro)
    #   keep_bools = tf.cast( bern.sample(self.X.shape), tf.bool )
    #   self.mask = tf.logical_or( tf.cast(self.mask, tf.bool) , keep_bools)
    #   self.mask = tf.cast(self.mask, tf.float32)

    # mask_decoder_out = self.decoder_out * t_mask
    # total_valid_nums = tf.reduce_sum(t_mask)

    tf.summary.histogram("encoder_out", self.encoder_out)
    tf.summary.histogram("decoder_out", self.decoder_out)
    # tf.summary.scalar("cal_loss_zeros_nums", total_valid_nums - origin_nums)
    # tf.summary.scalar("origin_valid_nums", origin_nums)
    # tf.summary.scalar("total_valid_nums", total_valid_nums)


    # # ##cross entropy
    # entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=mask_decoder_out, name="loss")
    # self.loss = tf.reduce_sum(entropy) / total_valid_nums

    # self.loss = tf.reduce_mean(tf.reduce_sum(entropy, reduction_indices=[1]))

    # # Define loss and optimizer, minimize the squared error

    mse = tf.reduce_mean( tf.boolean_mask(tf.pow(self.X - self.decoder_out, 2),tf.logical_not(self.mask)))
    self.mse_loss = mse / 2
   

    self.sparse_loss = tf.reduce_mean(tf.boolean_mask(tf.abs(self.decoder_out),self.mask))
    self.rank_loss = tf.reduce_sum(tf.svd(self.decoder_out, compute_uv=False))


    self.loss = self.mse_loss + self.gamma * self.sparse_loss + self.beta * self.rank_loss
    # self.loss = self.mse_loss + self.gamma * self.sparse_loss

    tf.summary.scalar("train_loss", self.loss)
    tf.summary.scalar("mse_loss", self.mse_loss)
    tf.summary.scalar("sparse_loss", self.sparse_loss)
    tf.summary.scalar("nuclear_loss", self.rank_loss)

    with tf.name_scope('optimizer'):
      # Gradient Descent
      # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      # Op to calculate every variable gradient
      grads = tf.gradients(self.loss, tf.trainable_variables())
      grads = list(zip(grads, tf.trainable_variables()))
      # Op to update all variables according to their gradient
      self.apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

    # Create summaries to visualize weights
    for var in tf.trainable_variables():
      tf.summary.histogram(var.name, var)

    # Summarize all gradients
    for grad, var in grads:
      tf.summary.histogram(var.name + '/gradient', grad)

    # Merge all summaries into a single op
    self.merged_summary_op = tf.summary.merge_all()

    self.saver = tf.train.Saver(max_to_keep=1)

    '''
      `max_to_keep` indicates the maximum number of recent checkpoint files to
      keep.  As new files are created, older files are deleted.  If None or 0,
      all checkpoint files are kept.  Defaults to 5 (that is, the 5 most recent
      checkpoint files are kept.)
    '''

  def encoder(self, input):

    with tf.variable_scope("encoder"):

      out = Dense(self.feature_num // 4, activation="relu")(input)
      if self.dropout > 0.0:
        out = keras.layers.Dropout(self.dropout)(out)
      out = Dense(self.feature_num // 16, activation="relu")(out)
      out = Dense(self.feature_num // 32)(out)
      out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

      # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
      # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
      # out = self.activation(out)

      #(None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
    return out

  def decoder(self, input):

    with tf.variable_scope("decoder") as D:

      # out = Dropout(0.2)(input)
      out = Dense(self.feature_num // 16, activation="relu")(input)
      out = Dense(self.feature_num // 4, activation="relu")(out)
      # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)

      if self.dropout > 0.0:
        out = keras.layers.Dropout(self.dropout)(out)
      out = Dense(self.feature_num, kernel_regularizer=regularizers.l2(0.01) )(out)

      out = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")(out)

      # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
      # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
      # out = self.activation(out)

      #(None, fe // 9) -> (None, fe // 3) -> (None, fe)
    return out

  def encoder_bn(self, input):
    with tf.variable_scope("encoder_bn"):
      out = Dense(self.feature_num // 4, activation="linear")(input)
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.activations.relu(out)
      if self.dropout < 1.0:
        out = keras.layers.Dropout(self.dropout)(out)
      out = Dense(self.feature_num // 16, activation="linear")(out)
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.activations.relu(out)
      out = Dense(self.feature_num // 32, activation="linear")(out)
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

      # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
      # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
      # out = self.activation(out)

      # (None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
    return out

  def decoder_bn(self, input):
    with tf.variable_scope("decoder_bn"):
      # out = Dropout(0.2)(input)
      out = Dense(self.feature_num // 16, activation="linear")(input)
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.activations.relu(out)
      out = Dense(self.feature_num // 4, activation="linear")(out)
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.activations.relu(out)
      # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)

      if self.dropout < 1.0:
        out = keras.layers.Dropout(self.dropout)(out)
      out = Dense(self.feature_num, activation="linear", kernel_regularizer=regularizers.l2(0.01))(out)
      # out = batch_norm(out, is_training=self.is_training)
      # out = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")(out)

      # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
      # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
      # out = self.activation(out)

      # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
    return out


  def predict_tmp(self, sess, step, dataset, config):
    print("testing for {}th...".format(step))
    dataset.reset()
    decoder_out_list, encoder_out_list = [], []
    mask_data = []
    while (1):
      batch_data = dataset.next()
      if batch_data is None:
        break
      mask = (batch_data == 0.0)
      # mask = np.float32(mask)
      mask_data.append(mask)
      # keep_bools = np.float32( np.zeros_like(batch_data) )

      decoder_out, encoder_out = sess.run([self.decoder_out, self.encoder_out], feed_dict={self.X: batch_data,
                                                        self.mask: mask,self.is_training:False,K.learning_phase(): 0})
      decoder_out_list.append(decoder_out)
      encoder_out_list.append(encoder_out)
    decoder_out = np.reshape(np.concatenate(decoder_out_list, axis=0), (-1, dataset.feature_nums))
    encoder_out = np.reshape(np.concatenate(encoder_out_list, axis=0), (-1, self.feature_num // 32))
    mask_data = np.reshape(np.concatenate(mask_data, axis=0), (-1, dataset.feature_nums))

    mask_data = np.float32(mask_data)
    decoder_out = mask_data * decoder_out +  dataset.data ##missing value now is completed, other values remain same
    # rev_normal_predict_data = data_preprocess.reverse_normalization(decoder_out, config.normal_factor) #reverse normalization

    df = pd.DataFrame(decoder_out, columns=dataset.columns)

    if os.path.exists(config.outDir) == False:
      os.makedirs(config.outDir)
    outDir = os.path.join(config.outDir, self.model_name)
    if os.path.exists(outDir) == False:
      os.makedirs(outDir)
    outPath = os.path.join(outDir, "{}.complete".format(self.model_name))
    if config.plot_complete:
      plot.plot_complete(pd.DataFrame(dataset.data, columns=dataset.columns), df, outPath.replace("complete", "pdf"), onepage=True)

    df.to_csv(outPath, index=None)

    print("save complete data from {} to {}".format(config.train_datapath, outPath))
    # pd.DataFrame(rev_normal_predict_data, columns=dataset.columns).to_csv(outPath.replace(".complete", ".revnormal"),index=None)
    # print("save rev normal data to {}".format(outPath.replace(".complete", ".revnormal")))

    pd.DataFrame(encoder_out).to_csv(outPath.replace(".complete", ".encoder.out"))

  def train(self, config):

    begin = time.clock()
    dataset = DataSet(config.train_datapath, config.batch_size)
    test_dataset = DataSet(config.train_datapath, config.batch_size, shuffle=False, onepass=True)
    # dataset = dataset
    # test_dataset = dataset
    create_conf = self.create_conf
    steps = dataset.steps * config.epoch
    print("total {} steps...".format(steps))

    # sample_dirs = os.path.join("samples", self.model_name)

    # for dir in [sample_dirs, log_dirs]:
    #   if os.path.exists(dir) == False:
    #     os.makedirs(dir)

    # gpu_conf = tf.ConfigProto()
    # gpu_conf.gpu_options.per_process_gpu_memory_fraction = config.gpu_ratio

    with tf.Session() as session:

      log_dirs = os.path.join("./logs", self.model_name)
      if os.path.exists(log_dirs) == False:
        os.makedirs(log_dirs)

      load_model_dir = os.path.join('./backup', self.model_name)
      if config.load_checkpoint and os.path.exists(load_model_dir):
        self.load(session, load_model_dir)
      elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)

      if config.load_checkpoint is False and os.path.exists(log_dirs):
        shutil.rmtree(log_dirs)
        os.makedirs(log_dirs)

      self.writer = tf.summary.FileWriter(log_dirs, session.graph)

      tf.global_variables_initializer().run()

      # sample_batch = dataset.sample_batch()
      # sample_mask = np.float32(sample_batch > 0.0)
      # sample_path = os.path.join(sample_dirs, "{}.sample".format(self.model_name))
      # pd.DataFrame(sample_batch).to_csv(sample_path, index=False)

      for step in range(steps+1):

        batch_data = dataset.next()
        mask = (batch_data == 0.0)
        # mask = np.float32(mask)


        if step % config.save_freq_steps != 0:
          _, loss, mse_loss, sparse_loss, rank_loss = session.run( [self.apply_grads, self.loss, self.mse_loss, self.sparse_loss, self.rank_loss],
                                           feed_dict={ self.X: batch_data, self.mask: mask,
                                                       self.is_training: True, K.learning_phase(): 1})
        else:
          _, summary_str, loss, mse_loss, sparse_loss, rank_loss = session.run( [self.apply_grads, self.merged_summary_op,
                                                                      self.loss, self.mse_loss, self.sparse_loss, self.rank_loss],
                                             feed_dict={ self.X: batch_data, self.mask: mask, self.is_training:True,
                                                        K.learning_phase(): 1})

        if step % config.log_freq_steps == 0:
          print("step {}th, loss: {}, mse_loss: {}, sparse_loss: {}, rank_loss: {}".format(step, loss, mse_loss, sparse_loss, rank_loss))

        if step % config.save_freq_steps == 0:
          self.writer.add_summary(summary_str, step)
          # save_dir = os.path.join(config.checkpoint_dir, self.model_name)
          self.save(session, load_model_dir, step)

      self.predict_tmp(session, steps, test_dataset, config)

    end = time.clock()
    print("training {} cost time: {} mins".format(self.model_name, (end - begin) / 60.0))


  def save(self, sess, save_dir, step):
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)

      self.saver.save(sess,
                      os.path.join(save_dir, self.model_name),
                      global_step=step)

  def load(self, sess, checkpoint_dir):
      print(" [*] Reading checkpoints...")

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
          self.saver.restore(sess, ckpt.model_checkpoint_path)
          return True
      else:
          return False
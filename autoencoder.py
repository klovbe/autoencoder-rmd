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

  def __init__(self, feature_num, hidden_size=None, dropout = None,
               learning_rate=0.001, activation="relu", model_name="auto_encoder", **kwargs):

    self.feature_num = feature_num
    if hidden_size is None:
      hidden_size = feature_num // 3
    self.hidden_size = hidden_size
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
    self.mask = tf.placeholder(tf.float32, [None, self.feature_num], name="mask")
    self.keep_bools = tf.placeholder(tf.float32, [None, self.feature_num], name="keep_bools")

    # self.encoder_out = self.encoder(self.X)  # through activation
    # self.decoder_out = self.decoder(self.encoder_out)  # must not through activation


    self.encoder_out = self.encoder_bn(self.X) #through activation
    self.decoder_out = self.decoder_bn(self.encoder_out)  #must not through activation

    origin_nums = tf.reduce_sum(self.mask)

    t_mask = tf.logical_or( tf.cast(self.mask, tf.bool) , tf.cast(self.keep_bools, tf.bool) )
    t_mask = tf.cast(t_mask, tf.float32)

    # if mis_pro is not None:
    #   bern = distributions.Bernoulli(1 - mis_pro)
    #   keep_bools = tf.cast( bern.sample(self.X.shape), tf.bool )
    #   self.mask = tf.logical_or( tf.cast(self.mask, tf.bool) , keep_bools)
    #   self.mask = tf.cast(self.mask, tf.float32)

    mask_decoder_out = self.decoder_out * t_mask
    total_valid_nums = tf.reduce_sum(t_mask)

    tf.summary.histogram("encoder_out", self.encoder_out)
    tf.summary.histogram("decoder_out", self.decoder_out)
    tf.summary.scalar("cal_loss_zeros_nums", total_valid_nums - origin_nums)
    tf.summary.scalar("origin_valid_nums", origin_nums)
    tf.summary.scalar("total_valid_nums", total_valid_nums)


    # # ##cross entropy
    # entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=mask_decoder_out, name="loss")
    # self.loss = tf.reduce_sum(entropy) / total_valid_nums

    # self.loss = tf.reduce_mean(tf.reduce_sum(entropy, reduction_indices=[1]))

    # # Define loss and optimizer, minimize the squared error

    mask_mse = tf.reduce_sum( tf.pow(self.X - mask_decoder_out, 2) )
    self.loss = mask_mse / total_valid_nums

    tf.summary.scalar("train_loss",self.loss)

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
      out = batch_norm(out, is_training=self.is_training)
      out = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")(out)

      # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
      # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
      # out = self.activation(out)

      # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
    return out

  def train_test_mnist(self, config):

    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      loaded = False
      if config.load_checkpoint and os.path.exists(load_model_dir):
        self.load(sess, config.checkpoint_dir)
        loaded = True

      elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)

      # Run the initializer
      tf.global_variables_initializer().run()

      if not loaded:
        # Training
        for i in range(1, config.steps + 1):

          # Prepare Data
          # Get the next batch of MNIST data (only images are needed, not labels)
          batch_x, _ = mnist.train.next_batch(config.batch_size)
          mask = np.ones_like(batch_x)

          # Run optimization op (backprop) and cost op (to get loss value)
          _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.mask: mask})

          # Display logs per step
          if i % config.log_freq_steps == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

          if i % config.save_freq_steps == 0:
            save_dir = os.path.join(config.checkpoint_dir, self.model_name)
            self.save(sess, save_dir, i)

      # Testing
      # Encode and decode images from test set and visualize their reconstruction.
      n = 4
      canvas_orig = np.empty((28 * n, 28 * n))
      canvas_recon = np.empty((28 * n, 28 * n))
      for i in range(n):
          # MNIST test set
          batch_x, _ = mnist.test.next_batch(n)
          mask = np.ones_like(batch_x)
          # Encode and decode the digit image
          g = sess.run(self.decoder_out, feed_dict={self.X: batch_x, self.mask: mask})

          # Display original images
          for j in range(n):
              # Draw the original digits
              canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                  batch_x[j].reshape([28, 28])
          # Display reconstructed images
          for j in range(n):
              # Draw the reconstructed digits
              canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                  g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()


  def predict_tmp(self, sess, step, dataset, config):
    print("testing for {}th...".format(step))
    dataset.reset()
    decoder_out_list, encoder_out_list = [], []
    mask_data = []
    while (1):
      batch_data = dataset.next()
      if batch_data is None:
        break
      mask = (batch_data > 0.0)
      mask = np.float32(mask)
      mask_data.append(mask)
      keep_bools = np.float32( np.zeros_like(batch_data) )

      decoder_out, encoder_out = sess.run([self.decoder_out, self.encoder_out], feed_dict={self.X: batch_data, self.mask: mask,
                                                                                           self.is_training:False,self.keep_bools:
                                                                                             keep_bools,
                                                                              K.learning_phase(): 0})
      decoder_out_list.append(decoder_out)
      encoder_out_list.append(encoder_out)
    decoder_out = np.reshape(np.concatenate(decoder_out_list, axis=0), (-1, dataset.feature_nums))
    encoder_out = np.reshape(np.concatenate(encoder_out_list, axis=0), (-1, self.feature_num // 32))
    mask_data = np.reshape(np.concatenate(mask_data, axis=0), (-1, dataset.feature_nums))

    decoder_out = (1.0 - mask_data) * decoder_out +  dataset.data ##missing value now is completed, other values remain same
    rev_normal_predict_data = data_preprocess.reverse_normalization(decoder_out, config.normal_factor) #reverse normalization

    df = pd.DataFrame(decoder_out, columns=dataset.columns)

    if os.path.exists(config.outDir) == False:
      os.makedirs(config.outDir)
    outDir = os.path.join(config.outDir, self.model_name)
    if os.path.exists(outDir) == False:
      os.makedirs(outDir)
    outPath = os.path.join(outDir, "{}.infer.complete".format(self.model_name))
    if config.plot_complete:
      plot.plot_complete(pd.DataFrame(dataset.data, columns=dataset.columns), df, outPath.replace("infer.complete", "pdf"), onepage=True)

    df.to_csv(outPath, index=None)

    print("save complete data from {} to {}".format(config.infer_complete_datapath, outPath))
    pd.DataFrame(rev_normal_predict_data, columns=dataset.columns).to_csv(outPath.replace(".complete", ".revnormal"),index=None)
    print("save rev normal data to {}".format(outPath.replace(".complete", ".revnormal")))

    pd.DataFrame(encoder_out).to_csv(outPath.replace(".infer.complete", ".encoder.out"))

  def train(self, config):

    begin = time.clock()
    dataset = DataSet(config.train_datapath, config.batch_size)
    test_dataset = DataSet(config.infer_complete_datapath, config.batch_size, onepass=True)
    create_conf = self.create_conf
    truly_mis_pro = create_conf.get("truly_mis_pro")
    random_mask_path = config.random_mask_path
    steps = dataset.steps * config.epoch
    # print("random_mask_path:", random_mask_path)
    if random_mask_path != "None":
      mask_probs = pd.read_csv(random_mask_path, index_col=0).transpose().values
    print("total {} steps...".format(steps))

    sample_dirs = os.path.join("samples", self.model_name)
    log_dirs = os.path.join("logs", self.model_name)

    for dir in [sample_dirs, log_dirs]:
      if os.path.exists(dir) == False:
        os.makedirs(dir)

    # gpu_conf = tf.ConfigProto()
    # gpu_conf.gpu_options.per_process_gpu_memory_fraction = config.gpu_ratio

    with tf.Session() as session:

      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      if config.load_checkpoint and os.path.exists(load_model_dir):
        self.load(session, load_model_dir)
      elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)

      if config.load_checkpoint is False and os.path.exists(log_dirs):
        shutil.rmtree(log_dirs)
        os.makedirs(log_dirs)

      self.writer = tf.summary.FileWriter(log_dirs, session.graph)

      tf.global_variables_initializer().run()

      sample_batch = dataset.sample_batch()
      sample_mask = np.float32(sample_batch > 0.0)
      sample_path = os.path.join(sample_dirs, "{}.sample".format(self.model_name))
      pd.DataFrame(sample_batch).to_csv(sample_path, index=False)

      for step in range(steps):

        batch_data = dataset.next()
        mask = (batch_data > 0.0)
        mask = np.float32(mask)

        ##mask = keep_bools | mask
        if random_mask_path != "None":
          probs = np.random.uniform(0.0, 1.0, batch_data.shape)
          xp, yp = probs.shape
          mask_probs_sub = mask_probs[:xp, :yp]
          keep_bools = (mask_probs_sub >= probs)
        else:
          if truly_mis_pro <= 0.0:
            keep_bools = np.zeros_like(batch_data)
          else:
            zero_nums = np.sum( batch_data == 0.0 )
            q = np.random.binomial(1, 1.0 - truly_mis_pro, (zero_nums) )
            keep_bools = np.zeros_like(batch_data)
            keep_bools[ np.where(batch_data == 0.0) ] = q

        keep_bools = np.float32(keep_bools)

        if step % config.save_freq_steps != 0:
          _, loss = session.run( [self.apply_grads, self.loss],
                                           feed_dict={ self.X: batch_data, self.mask: mask, self.keep_bools: keep_bools,
                                                       self.is_training: True,K.learning_phase(): 1})
        else:
          _, summary_str, loss = session.run( [self.apply_grads, self.merged_summary_op, self.loss],
                                             feed_dict={ self.X: batch_data, self.mask: mask,
                                                         self.keep_bools: keep_bools, self.is_training:True,
                                                        K.learning_phase(): 1
                                                         }
                                              )

        if step % config.log_freq_steps == 0:
          print("step {}th, loss: {}".format(step, loss))

        # if step % config.test_freq_steps == 0:
        #   self.predict_tmp(session, step, test_dataset, config)

        if step % config.save_freq_steps == 0:
          self.writer.add_summary(summary_str, step)
          save_dir = os.path.join(config.checkpoint_dir, self.model_name)
          self.save(session, save_dir, step)

      self.predict_tmp(session, steps, test_dataset, config)

    end = time.clock()
    print("training {} cost time: {} mins".format(self.model_name, (end - begin) / 60.0))

  def predict(self, config):
    dataset = DataSet(config.infer_complete_datapath, batch_size=config.batch_size,shuffle=False, onepass=True)
    predict_data = []
    with tf.Session() as sess:
      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      isLoaded = self.load(sess, load_model_dir)
      assert (isLoaded)
      # try:
      #   tf.global_variables_initializer().run()
      # except:
      #   tf.initialize_all_variables().run()
      while (1):
        batch_data = dataset.next()
        if batch_data is None:
          break
        mask = (batch_data > 0.0)
        mask = np.float32(mask)
        predicts = sess.run(self.decoder_out, feed_dict={self.X: batch_data, self.mask: mask, self.is_training:False, K.learning_phase(): 0})
        predict_data.append(predicts)
    predict_data = np.reshape(np.concatenate(predict_data, axis=0), (-1, dataset.feature_nums))
    df = pd.DataFrame(predict_data, columns=dataset.columns)
    if os.path.exists(config.outDir) == False:
      os.makedirs(config.outDir)
    outPath = os.path.join(config.outDir, "{}.infer.complete".format(self.model_name))
    df.to_csv(outPath, index=None)
    print("save complete data from {} to {}".format(config.infer_complete_datapath, outPath))

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
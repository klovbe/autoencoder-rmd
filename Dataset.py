#encoding=utf-8
import os
import sys
import codecs
import numpy as np
import pandas as pd

class DataSet:

  def __init__(self, path, batch_size=128, shuffle=True, onepass=False, nrows=None):
    print("make dataset from {}...".format(path))
    data = pd.read_csv(path, sep=",", nrows=nrows)
    self.columns = list(data.columns)
    data = np.float32(data.values)
    self.path = path
    self.data = data
    self.samples, self.feature_nums = data.shape
    self.cnt = 0
    self.batch_counter = 0
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.onepass = onepass
    print("batch_size is {}, have {} samples, {} features, step nums is {}".format(batch_size, self.samples,
                                                                                   self.feature_nums, self.steps))
    print("make dataset end")

  def next(self):

    batch_size = self.batch_size

    if self.cnt >= self.samples and self.onepass is True:  # for infer mode
      return None

    if self.cnt + batch_size >= self.samples:
      if self.onepass:  # if last pass piece, make batch_data
        batch_data = self.data[self.cnt:]
        self.cnt = self.samples
        print("the last batch, shape is {}...".format(batch_data.shape))
        return batch_data

      self.cnt = 0
      self.shuffle_data()

    be, en = self.cnt, min(self.samples, self.cnt + batch_size)
    #         yield data[be, en]
    batch_data = self.data[be: en]
    self.cnt = (self.cnt + batch_size) % self.samples
    self.batch_counter += 1
    print("getting {}th batch end".format(self.batch_counter))
    return batch_data

  def sample_batch(self):
    index = np.random.choice(list(range(self.samples)), self.batch_size)
    return self.data[index]

  def shuffle_data(self):
    np.random.shuffle(self.data)

  def reset(self):
    self.cnt = 0

  @property
  def steps(self):
    return self.samples // self.batch_size


if __name__ == "__main__":

  # dataset = DataSet()
  print("xx")
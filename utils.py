#encoding=utf-8
import numpy as np
import codecs
import sys
import pandas as pd

def MaskData(batch_data, missing_val):
  mask = np.ones_like(batch_data)
  mask[ np.equal(batch_data, missing_val) ] = 0
  return mask

def get_mask_probs(datapath):
  df = pd.read_csv(datapath, index_col=0).transpose()
  mask_probs = np.random.uniform(0.0, 1.0, df.shape)


def restore_data(data):
  pass

if __name__ == "__main__":
  q = MaskData([[1,2,3,0],[0,0,1,2]], 0)
  print(q)
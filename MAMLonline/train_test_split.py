import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics


def train_test_split_tdnn(x_seq, y_seq, K):    
    data_length = x_seq.shape[0]
    data_id     = np.arange(0, data_length)
    drift = 1200
    train_id     = data_id[int(drift):int(K+drift)]
    test_id    = data_id[int(data_length*0.5)::]
    
    x_train     = x_seq[train_id]
    y_train     = y_seq[train_id]
    x_test      = x_seq[test_id]
    y_test      = y_seq[test_id]
    
    return x_train, y_train, x_test, y_test
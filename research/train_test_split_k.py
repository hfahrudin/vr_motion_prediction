import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics


def train_test_split_k(x_seq, y_seq, K):    
    data_length = x_seq.shape[0]
    data_id     = np.arange(0, data_length)
    drift = 1200
    data_id = data_id[int(drift)::]
    
    train_id = data_id[0:int(data_length*0.5)]
    np.random.shuffle(train_id)
    k_id = train_id[0:K]
    test_id    = data_id[int(data_length*0.5)::]
    
    xtest_train     = x_seq[k_id]
    ytest_train     = y_seq[k_id]
    xtest_test      = x_seq[test_id]
    ytest_test      = y_seq[test_id]
    
    return xtest_train, ytest_train, xtest_test, ytest_test
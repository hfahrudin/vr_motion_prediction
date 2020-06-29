import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math
import tensorflow as tf
#from scipy import signal
from sklearn import preprocessing, metrics


def train_test_split_meta1(x_seq, y_seq, K):    
    data_length = x_seq.shape[0]
    data_id     = np.arange(0, data_length)
    drift = 1200
    data_id = data_id[int(drift)::]
    
    train_id = data_id[0:int((data_length-drift)*0.5)]
    test_id    = data_id[int((data_length-drift)*0.5)::]
    
    train1 = train_id[0:int(K)]
    train2 = train_id[int(K):int(2*K)]
    train3 = train_id[int(2*K):int(3*K)]
    train4 = train_id[int(3*K):int(4*K)]
    train5 = train_id[int(4*K):int(5*K)]

    test1 = test_id[0:int(K)]
    test2 = test_id[int(K):int(2*K)]
    test3 = test_id[int(2*K):int(3*K)]

    test4 = test_id[int(3*K):int(4*K)]
    test5 = test_id[int(4*K):int(5*K)]

    xtest_train     = [tf.convert_to_tensor(x_seq[train1]), tf.convert_to_tensor(x_seq[train2]), tf.convert_to_tensor(x_seq[train3]), tf.convert_to_tensor(x_seq[train4]), tf.convert_to_tensor(x_seq[train5]),tf.convert_to_tensor(x_seq[test1]), tf.convert_to_tensor(x_seq[test2]), tf.convert_to_tensor(x_seq[test3])]
    ytest_train     = [tf.convert_to_tensor(y_seq[train1]), tf.convert_to_tensor(y_seq[train2]), tf.convert_to_tensor(y_seq[train3]), tf.convert_to_tensor(y_seq[train4]), tf.convert_to_tensor(y_seq[train5]),tf.convert_to_tensor(y_seq[test1]), tf.convert_to_tensor(y_seq[test2]), tf.convert_to_tensor(y_seq[test3])]
    xtest_test      = [ tf.convert_to_tensor(x_seq[test4]), tf.convert_to_tensor(x_seq[test5])]
    ytest_test      = [ tf.convert_to_tensor(y_seq[test4]), tf.convert_to_tensor(y_seq[test5])]
    
    return xtest_train, ytest_train, xtest_test, ytest_test
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


def train_test_split_meta(x_seq, y_seq, K, N):    
    data_length = x_seq.shape[0]
    data_id     = np.arange(0, data_length)
    drift = 1200
    data_id = data_id[int(drift)::]
    
    train_id = data_id[0:int((data_length-drift)*0.5)]
    test_id    = data_id[int((data_length-drift)*0.5)::]
    
    train1 = train_id[0:int(K)]
    train2 = train_id[int(K):int(2*K)]
    
    test1 = test_id[0:int(N)]
    test2 = test_id[int(N):int(2*N)]

    xtask_train     = x_seq[train1]
    ytask_train     = y_seq[train1]
    xtask_val     = x_seq[train2]
    ytask_val     = y_seq[train2]
    
    dtrainx      = x_seq[test1]
    dtrainy      = y_seq[test1]
    
    dtestx      = x_seq[test2]
    dtesty      = y_seq[test2]
    return xtask_train, ytask_train,xtask_val, ytask_val, dtrainx, dtrainy, dtestx, dtesty
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:29:55 2020

@author: WFS , HF
"""



import argparse
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
from scipy import signal
import skinematics as skin
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
#from scipy import signal
from taskcaller import taskcaller
from taskcaller_train1 import taskcaller_train1
import random
# modularized library import
from train_test_split_k import train_test_split_k
from rms import rms    
from copy import copy
from datetime import datetime
from maml import *

from math import pi
from math import cos
from math import floor

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

dtype="float64"
tf.keras.backend.set_floatx(dtype)
print('Python version: ', sys.version)
print('Tensorflow version: ', tf.__version__)

'''
#####   SYSTEM INITIALIZATION    #####
'''
S=30
# np.random.seed(S)
# random.seed(S)
# tf.random.set_seed(S)

system_rate = 60
k_train = 100
x_train1, t_train1, x_val1, t_val1,input_nm, target_nm, data_length, DELAY_SIZE, train_eule_data, anticipation_size, train_time_data = taskcaller_train1('../dataset/trainingtask1.csv', system_rate, k_train)
x_train2, t_train2, x_val2, t_val2,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask2.csv', system_rate, k_train)
x_train3, t_train3, x_val3, t_val3,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask3.csv', system_rate, k_train)
x_train4, t_train4, x_val4, t_val4, _, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask4.csv', system_rate, k_train)
x_train5, t_train5, x_val5, t_val5,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask5.csv', system_rate, k_train)
x_train6, t_train6, x_val6, t_val6,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask6.csv', system_rate, k_train)
x_train7, t_train7, x_val7, t_val7,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask7.csv', system_rate, k_train)
x_train8, t_train8, x_val8, t_val8,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask8.csv', system_rate, k_train)
x_train9, t_train9, x_val9, t_val9,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask9.csv', system_rate, k_train)
x_train10, t_train10, x_val10, t_val10,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask9.csv', system_rate, k_train)
#x_seq10, t_seq10, _, _,_, _, _, _, _, _, _ = taskcaller('trainingtask10.csv', system_rate, k)


traintaskx = [x_train1 , x_train2 , x_train3 , x_train4,x_train5,x_train6,x_train7,x_train8,x_train9,x_train10]
traintaskt = [t_train1 , t_train2 , t_train3 , t_train4,t_train5,t_train6,t_train7, t_train8, t_train9,t_train10]

valtaskx = [x_val1,x_val2,x_val3,x_val4,x_val5,x_val6,x_val7,x_val8,x_val9,x_val10]
valtaskt = [t_val1,t_val2,t_val3,t_val4,t_val5,t_val6,t_val7,t_val8,t_val9,t_val10]

numberoftask = len(traintaskx)





model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(DELAY_SIZE, input_nm)),
        tf.keras.layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, 
        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ])



TRAINED_MODEL_NAME = 'best_net'



def update_procedure(model,dtx, dty, lr = 0.001, grad_step =10):
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    all_loss = []
    for step in range (grad_step):
        total_loss = 0
        for i in range(len(dtx)):
            with tf.GradientTape() as update:
                _,loss = model_func(model, dtx[i], dty[i])
            gradient = update.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            total_loss+=loss
        all_loss.append(total_loss/len(dtx))
        print('Step{} : loss = {}'.format(step,total_loss/len(dtx)))
    return model, all_loss  

#Online Meta Learning FTML
ftml = model
threshold = 3.5
meta_step = 10
loss_ftml = []
total = []
all_eval_loss = []
all_train_loss = []
xtask_buffer = []
ttask_buffer = []
buffer_length = len(xtask_buffer)
for i in range (10):
    print("Task ", i)
    eval_task = []
    train_loss = []
    
    if len(xtask_buffer) >= 15:
        xtask_buffer = xtask_buffer[5:]
        ttask_buffer = ttask_buffer[5:] 

    xtask_buffer = xtask_buffer[buffer_length//2:]
    ttask_buffer = ttask_buffer[buffer_length//2:]
    buffer_length = len(xtask_buffer)
    for j in range (8):
        
        total_loss_task = 0
        dtstream_x = traintaskx[i][0:j+1]
        dtstream_t = traintaskt[i][0:j+1]
        
        if len(xtask_buffer) <   1:
            dtrainx = dvalx = dtstream_x
            dtraint = dvalt = dtstream_t
        elif len(xtask_buffer) == 1:
            dtrainx = dvalx = xtask_buffer 
            dtraint = dvalt = ttask_buffer
        else :
            dtrainx = xtask_buffer[0:buffer_length//2]
            dvalx = xtask_buffer[buffer_length//2:]
            dtraint = ttask_buffer[0:buffer_length//2]
            dvalt = ttask_buffer[buffer_length//2:]

        print("Meta Update")
        ftml, loss = train_maml(ftml, meta_step, dtrainx, dtraint, dvalx, dvalt)
        total_loss_task += sum(loss)/len(loss)
        print("Update Procedure")
        ftml, loss = update_procedure(ftml,dtstream_x, dtstream_t)
        total_loss_task = (total_loss_task + sum(loss)/len(loss))/2

        tmp_loss = 0
        for k in range(len(valtaskx[j])):
            _, loss = model_func(ftml, valtaskx[i][k], valtaskt[i][k])
            tmp_loss+=loss

        eval_loss = tmp_loss/2
        eval_task.append(eval_loss)
        train_loss.append(total_loss_task)

        print('Data stream Batch- {} : loss = {}'.format(j,eval_loss))
        # if eval_loss < threshold or j == 9:
        #     print("Training Finish")
        #     total.append(j+1)
        #     loss_ftml.append(eval_loss)
        #     break
    xtask_buffer+=dtstream_x
    ttask_buffer+=dtstream_t
    all_train_loss.append(train_loss)
    all_eval_loss.append(eval_task)

# plt.figure(1)   
# plt.title("MAE (Training)")   
# for item in all_train_loss:
#     plt.plot(item)
# plt.legend(["Task 1", "Task 2","Task 3","Task 4","Task 5","Task 6","Task 7","Task 8","Task 9","Task 10"], loc=(1.05, 0.5))
# plt.xlabel('Data Size (index*100)')
# plt.ylabel('MAE')
# plt.show()

# plt.figure(2)      
# plt.title("MAE (Eval)")
# for item in all_eval_loss:
#     plt.plot(item)
# plt.legend(["Task 1", "Task 2","Task 3","Task 4","Task 5","Task 6","Task 7","Task 8","Task 9","Task 10"], loc=(1.05, 0.5))
# plt.xlabel('Data Size (index*100)')
# plt.ylabel('MAE')
# plt.show()

# plt.figure(3)    
# plt.plot(total)  
# plt.title("Task Learning Efficiency")
# plt.xlabel('Task Index')
# plt.ylabel('Amount of data needed for loss threshold')
# plt.show()

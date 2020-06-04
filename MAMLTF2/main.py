# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:29:55 2020

@author: WFS
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
np.random.seed(S)
random.seed(S)
tf.random.set_seed(S)

system_rate = 60
k_train = 300
x_train1, t_train1, x_val1, t_val1,input_nm, target_nm, data_length, DELAY_SIZE, train_eule_data, anticipation_size, train_time_data = taskcaller_train1('../dataset/trainingtask1.csv', system_rate, k_train)
x_train2, t_train2, x_val2, t_val2,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask2.csv', system_rate, k_train)
x_train3, t_train3, x_val3, t_val3,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask3.csv', system_rate, k_train)
x_train4, t_train4, x_val4, t_val4, _, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask4.csv', system_rate, k_train)
x_train5, t_train5, x_val5, t_val5,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask5.csv', system_rate, k_train)
x_train6, t_train6, x_val6, t_val6,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask6.csv', system_rate, k_train)
x_train7, t_train7, x_val7, t_val7,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask7.csv', system_rate, k_train)
x_train8, t_train8, x_val8, t_val8,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask8.csv', system_rate, k_train)
x_train9, t_train9, x_val9, t_val9,_, _, _, _, _, _, _ = taskcaller_train1('../dataset/trainingtask9.csv', system_rate, k_train)
#x_seq10, t_seq10, _, _,_, _, _, _, _, _, _ = taskcaller('trainingtask10.csv', system_rate, k)


traintaskx = x_train1 + x_train2 + x_train3 + x_train4+x_train5+x_train6+x_train7+x_train8+x_train9
traintaskt = t_train1 + t_train2 + t_train3 + t_train4+t_train5+t_train6+t_train7+ t_train8+ t_train9

valtaskx = x_val1+x_val2+x_val3+x_val4+x_val5+x_val6+x_val7+x_val8+x_val9
valtaskt = t_val1+t_val2+t_val3+t_val4+t_val5+t_val6+t_val7+t_val8+t_val9

numberoftask = len(traintaskx)

#pretrain_x = tf.concat([x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, x_train9],0)
#pretrain_t = tf.concat([t_train1, t_train2, t_train3, t_train4, t_train5, t_train6, t_train7, t_train8, t_train9],0)

pretrain_x = traintaskx[0]
pretrain_t = traintaskt[0]
for i in range(numberoftask-1):
    pretrain_x = tf.concat([pretrain_x,traintaskx[i]],0)
    pretrain_t = tf.concat([pretrain_t,traintaskt[i]],0)

##placeholder only can be executed by disable eager execution
#tf.compat.v1.disable_eager_execution()

## Reset the whole tensorflow graphn
#tf.compat.v1.reset_default_graph()


model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(DELAY_SIZE, input_nm)),
        tf.keras.layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, 
        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ])


# Hyperparameter training
p_epochs = 100
epochs1 = 100
epochs = 50



# Variables
TRAINED_MODEL_NAME = 'best_net'


start_time = time.time()

_, losses_1= orderone(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt,inner_loop=1)
_, losses_3= orderone(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt,inner_loop=3)
_, losses_4= orderone(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt, inner_loop=6)
_, losses_5= train_maml(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt )

plt.plot(losses_1)
plt.plot(losses_3)
plt.plot(losses_4)
plt.plot(losses_5)
plt.xlabel("Adaptation steps")
plt.title("Mean Absolute Error Performance (MAML)")
plt.legend(["1 inner loop","3 inner loop", "5 inner loop","Regular"], loc=(0.5, 0.5))
plt.ylabel("MAE")
plt.show()

fin_time = time.time()
print('total time', fin_time - start_time)


def eval_func (model, optimizer, x, y, x_test, y_test, num_steps):
    newloss = []
    tensor_x_test, tensor_y_test = x_test, y_test
    
    if 0 in num_steps:
        logits, loss = model_func(model, tensor_x_test, tensor_y_test)
        newloss.append(loss)

    
    for step in range(1, np.max(num_steps) + 1):
        losst, model = train_batch(x, y, model, optimizer)
        logits, loss = model_func(model, tensor_x_test, tensor_y_test)
        newloss.append(loss)
        fit_res = np.array(logits)
    return fit_res, newloss

def eval_adaptation (model,system_rate, k_test, num_steps = (0, 1, 10), lr = 0.01):
#    total_loss = np.zeros(11)
    xtest_train,ytest_train,xtest_test,ytest_test, _, _, _, _, _, _, _ = taskcaller('trainingtask10.csv', system_rate, k_test)
    model_copy = copy_model(model, xtest_train)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    fit_res, newloss = eval_func(model_copy, optimizer, xtest_train, ytest_train, xtest_test, ytest_test, num_steps)
    loss_step = np.array(newloss)
    plt.figure()
    plt.plot(loss_step)
    plt.title("MAE Adaptation")
    plt.show()
    return fit_res,loss_step, ytest_test

k_test1 = 60
k_test2 = 300
k_test3 = 600
k_test4 = 3000

fit_res1,total_loss1, ytest_test1 = eval_adaptation(trainmaml1st, system_rate, k_test1)
fit_res2,total_loss2, ytest_test2 = eval_adaptation(trainmaml1st, system_rate, k_test2)
fit_res3,total_loss3, ytest_test3 = eval_adaptation(trainmaml1st, system_rate, k_test3)
fit_res4,total_loss4, ytest_test4 = eval_adaptation(trainmaml1st, system_rate, k_test4)

y_out1 = fit_res1
prediction1 = signal.savgol_filter(y_out1, 81, 3, axis=0)
y_out2 = fit_res2
prediction2 = signal.savgol_filter(y_out2, 81, 3, axis=0)
y_out3 = fit_res3
prediction3 = signal.savgol_filter(y_out3, 81, 3, axis=0)
y_out4 = fit_res4
prediction4 = signal.savgol_filter(y_out4, 81, 3, axis=0)

# Calculate Orientation Error
euler_o1 = ytest_test1[:]
euler_pred_ann1 = prediction1[:, :3]
euler_ann_err1 = np.abs(euler_pred_ann1[:-anticipation_size] - euler_o1[anticipation_size:])

euler_o2 = ytest_test2[:]
euler_pred_ann2 = prediction2[:, :3]
euler_ann_err2 = np.abs(euler_pred_ann2[:-anticipation_size] - euler_o2[anticipation_size:])

euler_o3 = ytest_test3[:]
euler_pred_ann3 = prediction3[:, :3]
euler_ann_err3 = np.abs(euler_pred_ann3[:-anticipation_size] - euler_o3[anticipation_size:])

euler_o4 = ytest_test4[:]
euler_pred_ann4 = prediction4[:, :3]
euler_ann_err4 = np.abs(euler_pred_ann4[:-anticipation_size] - euler_o4[anticipation_size:])

# Plot
timestamp_plot = train_time_data[DELAY_SIZE:-2*anticipation_size]
time_offset = timestamp_plot[0]
timestamp_plot = np.array(timestamp_plot)-time_offset

# Split error value
#euler_ann_err_train = euler_ann_err[: int(TRAIN_SIZE*data_length)] 
#euler_ann_err_test = euler_ann_err[int((1-TEST_SIZE)*data_length):] 

# calculate MAE error
#ann_mae_train = np.nanmean(np.abs(euler_ann_err_train), axis=0)
ann_mae_test1 = np.nanmean(np.abs(euler_ann_err1), axis=0)
ann_mae_test2 = np.nanmean(np.abs(euler_ann_err2), axis=0)
ann_mae_test3 = np.nanmean(np.abs(euler_ann_err3), axis=0)
ann_mae_test4 = np.nanmean(np.abs(euler_ann_err4), axis=0)

#print('Train')
#print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_train[0], ann_mae_train[1], ann_mae_train[2]))
## print('99% ANN training, {:.2f}, {:.2f}, {:.2f}'.format(final_train_99pitch, final_train_99roll, final_train_99yaw))

print('Test 60')
print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test1[0], ann_mae_test1[1], ann_mae_test1[2]))
print('Test 300')
print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test2[0], ann_mae_test2[1], ann_mae_test2[2]))
print('Test 600')
print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test3[0], ann_mae_test3[1], ann_mae_test3[2]))
print('Test 3000')
print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test4[0], ann_mae_test4[1], ann_mae_test4[2]))

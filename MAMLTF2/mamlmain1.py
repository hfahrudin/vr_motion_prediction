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
x_train1, t_train1, x_val1, t_val1,input_nm, target_nm, data_length, DELAY_SIZE, train_eule_data, anticipation_size, train_time_data = taskcaller_train1('trainingtask1.csv', system_rate, k_train)
x_train2, t_train2, x_val2, t_val2,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask2.csv', system_rate, k_train)
x_train3, t_train3, x_val3, t_val3,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask3.csv', system_rate, k_train)
x_train4, t_train4, x_val4, t_val4, _, _, _, _, _, _, _ = taskcaller_train1('trainingtask4.csv', system_rate, k_train)
x_train5, t_train5, x_val5, t_val5,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask5.csv', system_rate, k_train)
x_train6, t_train6, x_val6, t_val6,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask6.csv', system_rate, k_train)
x_train7, t_train7, x_val7, t_val7,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask7.csv', system_rate, k_train)
x_train8, t_train8, x_val8, t_val8,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask8.csv', system_rate, k_train)
x_train9, t_train9, x_val9, t_val9,_, _, _, _, _, _, _ = taskcaller_train1('trainingtask9.csv', system_rate, k_train)
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
#
#
## Reset the whole tensorflow graph
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

#def updateTargetModel(model, targetModel):
#    modelWeights       = model.trainable_weights
#    targetModelWeights = targetModel.trainable_weights
#    for i in range(len(targetModelWeights)):
#        targetModelWeights[i].assign(modelWeights[i])

#
def copy_model(model, x):
    '''
        To copy weight/variable model to the next model
    '''
    copied_model = tf.keras.models.clone_model(model)
    copied_model.forward = model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

# Hyperparameter training
p_epochs = 100
epochs1 = 100
epochs = 200
#learning_rate = 0.001
#optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate)
loss_function = tf.losses.MeanSquaredError()

def cosine_annealing(epoch, total_epoch, lrate_max):
    cos_val = (pi*epoch)/total_epoch
    lrate_val = (lrate_max/2)*(cos(cos_val)+1)
    return lrate_val

# Variables
TRAINED_MODEL_NAME = 'best_net'

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def tensor_to_np(tensor):
    return (np.array(obj) for obj in tensor)


#compile python code into graph

def model_func(model, x_train, t_train): #compute loss
    y_pred = model(x_train)
    loss = tf.losses.MeanSquaredError()(y_pred,t_train)
    return y_pred, loss
#@tf.function
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = (x,y)
    with tf.GradientTape() as tape:
        _, loss= model_func(model, tensor_x, tensor_y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, model

#@tf.function()
def train_maml (model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, lr_inner = 0.01, lr_outer_max = 0.02, log_step = 100):
    losses = []
    print("Training is starting")
    for epoch in range (epochs):
        total_loss = 0 
        outer_loss = 0
        lr_outer = cosine_annealing(epoch, epochs, lr_outer_max)
        opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
        with tf.GradientTape() as outer_tape: 
            for i in range (numberoftask): #for all dataset t with size i
                x = traintaskx[i]
                y = traintaskt[i] #convert dataset into tensor
                xval = valtaskx[i]
                yval = valtaskt[i]
                model(x)#forward pass to initialize weights
                #step 5
                with tf.GradientTape() as inner_tape:
                    _,inner_loss = model_func(model, x, y)
                #step 6
                gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
                model_copy = copy_model(model,x)
    #                print('model copy layer',len(model_copy.layers))
                k = 0
                for j in range(len(model_copy.layers)):
                    if j % 2 == 0:
                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients2[k]))
                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients2[k+1]))
                        k += 2
                y_pred, test_loss = model_func(model_copy, xval, yval)
                outer_loss += test_loss
                #step 8  
#        print('model 1 summ', model.summary())
#        print('model 2 summ', model_copy.summary())
        gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
#        print('gradient value', gradients)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss = outer_loss
        loss = total_loss / (len(traintaskx))
        losses.append(loss)
#        if epoch % log_step == 0 and epoch > 0:
        print('Step{} : loss = {}'.format(epoch,loss))
    return model, losses, model_copy

##model alternative original
#def train_maml (model, epochs, traintaskx, traintaskt, valtaskx, valtaskt, lr_inner = 0.01, lr_outer = 0.008, log_step = 100):
#    losses = []
#    print("Training is starting")
#    optimizer = keras.optimizers.Adamax(learning_rate=lr_outer)
#    for epoch in range (epochs):
#        total_loss = 0 
#        outer_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
#        for i in range (numberoftask): #for all dataset t with size i
#            x = traintaskx[i]
#            y = traintaskt[i] #convert dataset into tensor
#            xval = valtaskx[i]
#            yval = valtaskt[i]
#            model(x)#forward pass to initialize weights
#            with tf.GradientTape() as outer_tape: 
#                outer_tape.watch(outer_loss)
#                inner_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
#                #step 5
#                model_copy = copy_model(model,x)
#                with tf.GradientTape() as inner_tape:
#                    inner_tape.watch(model.trainable_variables)
#                    inner_tape.watch(inner_loss)
#                    _,inner_loss = model_func(model, x, y)
#                #step 6
#                gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
#    #            print('gradients2 value',gradients2)
#    #            print('model 1 summ', model.summary())
#    #            print('model 2 summ', model_copy.summary())
#    #            new_weights = model.get_weights()
#    #            for j in range(len(model_copy.get_weights())):
#    #                new_weights[j] = tf.subtract(new_weights[j], tf.multiply(0.01, gradients2[j]))
#    #            model_copy.set_weights(new_weights)
#                k = 0
#                for j in range(len(model_copy.layers)):
#                    if j % 2 == 0:
#                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients2[k]))
#                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients2[k+1]))
#                        k += 2
#    #            print('model 1 summ', model.summary())
#    #            print('model 2 summ', model_copy.summary())
#                y_pred, test_loss = model_func(model_copy, xval, yval)
#    #            print('Step: loss = {}'.format(test_loss))
#                #step 8  
#    #    print('model 1 summ', model.summary())
#    #    print('model 2 summ', model_copy.summary())
#            gradients = outer_tape.gradient(test_loss, model.trainable_variables)
#            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#            total_loss += test_loss
#        loss = total_loss / (numberoftask)
#        losses.append(loss)
#        print('Step{} : loss = {}'.format(epoch,loss))
#    return model, losses, model_copy

def order1st (model, epochs, traintaskx, traintaskt, valtaskx, valtaskt, lr_inner = 0.01, lr_outer_max = 0.01, log_step = 100):
    losses = []
    print("Training is starting")
    for epoch in range (epochs):
        total_loss = 0 
        outer_gradient =[]
        outer_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
        lr_outer = cosine_annealing(epoch, epochs, lr_outer_max)
        opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
        for i in range (numberoftask): #for all dataset t with size i
            x = traintaskx[i]
            y = traintaskt[i] #convert dataset into tensor
            xval = valtaskx[i]
            yval = valtaskt[i]
            
            with tf.GradientTape() as outer_tape: 
                outer_tape.watch(outer_loss)
                inner_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
                model(x)#forward pass to initialize weights
                #step 5
                model_copy = copy_model(model,x)
                updated_variables = model_copy.trainable_variables
                with tf.GradientTape() as inner_tape:
                    updated_variables = updated_variables
                    inner_tape.watch(inner_loss)
                    ypredin,inner_loss = model_func(model, x, y)
                gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
                k = 0
                for j in range(len(model_copy.layers)):
                    if j % 2 == 0:
                        model_copy.layers[j].kernel = updated_variables[k]  = tf.subtract(updated_variables[k], tf.multiply(0.01, gradients2[k]))
                        model_copy.layers[j].bias = updated_variables[k+1] = tf.subtract(updated_variables[k+1], tf.multiply(0.01, gradients2[k+1]))
                        k += 2
    
                y_pred, test_loss = model_func(model_copy, xval, yval)
                outer_loss += test_loss
    #            print('Step: loss = {}'.format(test_loss))
                #step 8  
    #    print('model 1 summ', model.summary())
    #    print('model 2 summ', model_copy.summary())
            gradients = outer_tape.gradient(test_loss, updated_variables)
            outer_gradient.append(gradients)
        #averaging gradient value
        zeros_grad = [tf.Variable(tf.zeros_like(gradients), trainable=False) for gradients in gradients]
        for i in range(len(outer_gradient)):
            for j in range(len(zeros_grad)):
                zeros_grad[j] = tf.add(zeros_grad[j],outer_gradient[i][j])
                
        for j in range(len(zeros_grad)):
            zeros_grad[j] = tf.divide(zeros_grad[j], len(outer_gradient))
        
        opt_outer.apply_gradients(zip(zeros_grad, model.trainable_variables))
        total_loss = outer_loss
        loss = total_loss / (len(traintaskx))
        losses.append(loss)
        print('Step{} : loss = {}'.format(epoch,loss))
    return model
#model aternative average   
#def train_maml (model, epochs, traintaskx, traintaskt, valtaskx, valtaskt, lr_inner = 0.01, lr_outer_max = 0.01, log_step = 100):
#    losses = []
#    print("Training is starting")
#    for epoch in range (epochs):
#        total_loss = 0 
#        outer_gradient =[]
#        lr_outer = cosine_annealing(epoch, epochs, lr_outer_max)
#        opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
#        outer_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
#        for i in range (numberoftask): #for all dataset t with size i
#            x = traintaskx[i]
#            y = traintaskt[i] #convert dataset into tensor
#            xval = valtaskx[i]
#            yval = valtaskt[i]
#            model(x)#forward pass to initialize weights
#            with tf.GradientTape() as outer_tape: 
#                outer_tape.watch(outer_loss)
#                inner_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
#                #step 5
#                model_copy = copy_model(model,x)
#                with tf.GradientTape() as inner_tape:
#                    inner_tape.watch(model.trainable_variables)
#                    inner_tape.watch(inner_loss)
#                    _,inner_loss = model_func(model, x, y)
#                #step 6
#                gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
#                k = 0
#                for j in range(len(model_copy.layers)):
#                    if j % 2 == 0:
#                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients2[k]))
#                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients2[k+1]))
#                        k += 2
#                y_pred, test_loss = model_func(model_copy, xval, yval)
#                outer_loss += test_loss
#
#            gradients = outer_tape.gradient(test_loss, model.trainable_variables)
#            outer_gradient.append(gradients)
#        #averaging gradient value
#        zeros_grad = [tf.Variable(tf.zeros_like(gradients), trainable=False) for gradients in gradients]
#        for i in range(len(outer_gradient)):
#            for j in range(len(zeros_grad)):
#                zeros_grad[j] = tf.add(zeros_grad[j],outer_gradient[i][j])
#                
#        for j in range(len(zeros_grad)):
#            zeros_grad[j] = tf.divide(zeros_grad[j], len(outer_gradient))
#        
#        opt_outer.apply_gradients(zip(zeros_grad, model.trainable_variables))
#        total_loss = outer_loss
#        loss = total_loss / (len(traintaskx))
#        losses.append(loss)
#        print('Step{} : loss = {}'.format(epoch,loss))
#    return model, losses,model_copy

#
#def train_maml (model, epochs, traintaskx, traintaskt, lr_inner = 0.01, lr_outer = 0.01, log_step = 100):
#    losses = []
#    start_time = time.time()
#    print("Training is starting")
#    optimizer = keras.optimizers.Adam(learning_rate=lr_outer)
#    for epoch in range (epochs):
#        total_loss = 0 
#        outer_loss = 0
#        for i in range (numberoftask): #for all dataset t with size i
#            x = traintaskx[i]
    
#            y = traintaskt[i] #convert dataset into tensor
#            model(x)#forward pass to initialize weights
#            with tf.GradientTape() as outer_tape: 
#                #step 5
#                with tf.GradientTape() as inner_tape:
#                    _,inner_loss = model_func(model, x, y)
#                #step 6
#                gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
#                model_copy = copy_model(model,x)
#    #                print('model copy layer',len(model_copy.layers))
##                new_weights = model.get_weights()
##                for j in range(len(model_copy.get_weights())):
##                    new_weights[j] = tf.subtract(new_weights[j], tf.multiply(0.01, gradients2[j]))
##                model_copy.set_weights(new_weights)
#                k = 0
#                for j in range(len(model_copy.layers)):
#                    if j % 2 == 0:
#                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients2[k]))
#                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients2[k+1]))
#                        k += 2
#    #            print('model 1 summ', model.summary())
#    #            print('model 2 summ', model_copy.summary())
#                y_pred, test_loss = model_func(model_copy, x, y)
#                outer_loss += test_loss
#                #step 8  
##        print('model 1 summ', model.summary())
##        print('model 2 summ', model_copy.summary())
#            gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
#    #        print('gradient value', gradients)
#            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#            total_loss = outer_loss
#        loss = total_loss / (len(traintaskx))
#        losses.append(loss)
##        if epoch % log_step == 0 and epoch > 0:
#        print('Step{} : loss = {}'.format(epoch,loss))
#    fin_time = time.time()
#    print('total time', fin_time - start_time)
#    return model, losses,model_copy
start_time = time.time()
#model1st = order1st(model, epochs1, traintaskx, traintaskt, valtaskx, valtaskt )
trainmaml, losses, model_copy = train_maml(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt )
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

fit_res1,total_loss1, ytest_test1 = eval_adaptation(trainmaml, system_rate, k_test1)
fit_res2,total_loss2, ytest_test2 = eval_adaptation(trainmaml, system_rate, k_test2)
fit_res3,total_loss3, ytest_test3 = eval_adaptation(trainmaml, system_rate, k_test3)
fit_res4,total_loss4, ytest_test4 = eval_adaptation(trainmaml, system_rate, k_test4)

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

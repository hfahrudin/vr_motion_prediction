# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:21:52 2020

@author: WFS
"""


import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
import sys
from scipy import signal
import matplotlib.pyplot as plt
#from scipy import signal
from taskcaller_train import taskcaller_train
import random
# modularized library import


from math import pi
from math import cos

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
n_eval = 100
xtask_train1, ytask_train1,xtask_val1,ytask_val1, dtrainx1, dtrainy1, dtestx1, dtesty1,input_nm, target_nm, data_length, DELAY_SIZE, train_eule_data, anticipation_size, train_time_data = taskcaller_train('trainingtask1.csv', system_rate, k_train, n_eval)
xtask_train2, ytask_train2,xtask_val2,ytask_val2, dtrainx2, dtrainy2, dtestx2, dtesty2,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask2.csv', system_rate, k_train, n_eval)
xtask_train3, ytask_train3,xtask_val3,ytask_val3, dtrainx3, dtrainy3, dtestx3, dtesty3,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask3.csv', system_rate, k_train, n_eval)
xtask_train4, ytask_train4,xtask_val4,ytask_val4, dtrainx4, dtrainy4, dtestx4, dtesty4, _, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask4.csv', system_rate, k_train, n_eval)
xtask_train5, ytask_train5,xtask_val5,ytask_val5, dtrainx5, dtrainy5, dtestx5, dtesty5,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask5.csv', system_rate, k_train, n_eval)
xtask_train6, ytask_train6,xtask_val6,ytask_val6, dtrainx6, dtrainy6, dtestx6, dtesty6,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask6.csv', system_rate, k_train, n_eval)
xtask_train7, ytask_train7,xtask_val7,ytask_val7, dtrainx7, dtrainy7, dtestx7, dtesty7,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask7.csv', system_rate, k_train, n_eval)
xtask_train8, ytask_train8,xtask_val8,ytask_val8, dtrainx8, dtrainy8, dtestx8, dtesty8,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask8.csv', system_rate, k_train, n_eval)
xtask_train9, ytask_train9,xtask_val9,ytask_val9, dtrainx9, dtrainy9, dtestx9, dtesty9,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask9.csv', system_rate, k_train, n_eval)
xtask_train10, ytask_train10,xtask_val10,ytask_val10, dtrainx10, dtrainy10, dtestx10, dtesty10,_, _, _, _, _, _, _ = taskcaller_train('../dataset/trainingtask10.csv', system_rate, k_train,n_eval)
#x_seq10, t_seq10, _, _,_, _, _, _, _, _, _ = taskcaller('trainingtask10.csv', system_rate, k)


traintaskx = [xtask_train1, xtask_train2, xtask_train3, xtask_train4,xtask_train5,xtask_train6,xtask_train7,xtask_train8,xtask_train9]
traintaskt = [ytask_train1, ytask_train2, ytask_train3, ytask_train4,ytask_train5,ytask_train6,ytask_train7, ytask_train8, ytask_train9]

valtaskx = [xtask_val1, xtask_val2, xtask_val3, xtask_val4,xtask_val5,xtask_val6,xtask_val7,xtask_val8,xtask_val9,xtask_val10]
valtaskt = [ytask_val1, ytask_val2, ytask_val3, ytask_val4,ytask_val5,ytask_val6,ytask_val7, ytask_val8, ytask_val9,ytask_val10]

#data task 1 will not use to make sure they are new datapoints
dtrainx = [dtrainx2,dtrainx3,dtrainx4,dtrainx5,dtrainx6,dtrainx7,dtrainx8,dtrainx9,dtrainx10]
dtrainy = [dtrainy2,dtrainy3,dtrainy4,dtrainy5,dtrainy6,dtrainy7,dtrainy8,dtrainy9,dtrainy10]

dtestx = [dtestx2,dtestx3,dtestx4,dtestx5,dtestx6,dtestx7,dtestx8,dtestx9,dtestx10]
dtesty = [dtesty2,dtesty3,dtesty4,dtesty5,dtesty6,dtesty7,dtesty8,dtesty9,dtesty10]

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
epochs = 100
#learning_rate = 0.001
#optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate)
loss_function = tf.losses.MeanSquaredError()

def cosine_annealing(epoch, total_epoch, lrate_max):
    cos_val = (pi*epoch)/total_epoch
    lrate_val = (lrate_max/2)*(cos(cos_val)+1)
    return lrate_val

TRAINED_MODEL_NAME = 'best_net'

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def tensor_to_np(tensor):
    return (np.array(obj) for obj in tensor)\

def model_func(model, x_train, t_train): #compute loss
    y_pred = model(x_train)
    loss = tf.losses.MeanSquaredError()(y_pred,t_train)
    return y_pred, loss

def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = (x,y)
    with tf.GradientTape() as tape:
        _, loss= model_func(model, tensor_x, tensor_y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, model

#meta update function
def train_maml (model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, lr_inner = 0.01, lr_outer_max = 0.01, log_step = 100):
    losses = []
    print("Training is starting")
    for epoch in range (epochs):
        total_loss = 0 
        outer_loss = 0
        lr_outer = cosine_annealing(epoch, epochs, lr_outer_max)
        opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
        with tf.GradientTape() as outer_tape: 
            for i in range (len(traintaskx)): #for all dataset t with size i
                x = traintaskx[i]
                y = traintaskt[i] #convert dataset into tensor
                xval = valtaskx[i]
                yval = valtaskt[i]
#                
#                x = tf.convert_to_tensor(traintaskx[i])
#                y = tf.convert_to_tensor(traintaskt[i]) #convert dataset into tensor
#                xval = tf.convert_to_tensor(valtaskx[i])
#                yval = tf.convert_to_tensor(valtaskt[i])
#                
                model(x)#forward pass to initialize weights
                #step 5
                with tf.GradientTape() as inner_tape:
                    y_in,inner_loss = model_func(model, x, y)
                #step 6
                gradients_in = inner_tape.gradient(inner_loss, model.trainable_variables)
                model_copy = copy_model(model,x)
    #                print('model copy layer',len(model_copy.layers))
                k = 0
                for j in range(len(model_copy.layers)):
                    if j % 2 == 0:
                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients_in[k]))
                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients_in[k+1]))
                        k += 2
                
                y_val, val_loss = model_func(model_copy, xval, yval)
                outer_loss += val_loss

        gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss = outer_loss
        loss = total_loss / (len(traintaskx))
        losses.append(loss)
        print('Step{} : loss = {}'.format(epoch,loss))
    return model

start_time = time.time()
#trainmaml, losses, model_copy = train_maml(model, epochs, traintaskx, traintaskt, valtaskx, valtaskt )
fin_time = time.time()
print('total time', fin_time - start_time)

#Update Procedure

def update_procedure(model,dtx, dty, lr = 0.01, grad_step =10):
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    for step in range (grad_step):
#        dtx = tf.convert_to_tensor(dtx)
#        dty = tf.convert_to_tensor(dty)
        model(dtx)
        with tf.GradientTape() as update:
            y_up,loss = model_func(model, dtx, dty)
        gradient = update.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return model

#Online Meta Learning FTML
ftml = model
threshold = 10
buffer_task_in = traintaskx[0]
buffer_task_out = traintaskt[0]

buffer_taskv_in = valtaskx[0]
buffer_taskv_out = valtaskt[0]

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
meta_step = 10
loss_ftml = []
point_ftml = []
#i = datastream
for i in range (len(traintaskx)):
    buffer_task_in = traintaskx[0:i+1]
    buffer_task_out = traintaskt[0:i+1]
    
    buffer_taskv_in = valtaskx[0:i+1]
    buffer_taskv_out = valtaskt[0:i+1]
    
    dtrain_in = dtrainx[i]
    dtrain_out = dtrainy[i]
    
    dtest_in = dtestx[i]
    dtest_out = dtesty[i]
    
    dtstream_in = []
    dtstream_out = []
    total_data = 0
    print('Task Streaming Number - {}'.format(i))
    
    for j in range (10):
        total_data += 30
        dtstream_in = dtrain_in[0:total_data,:,:]
        dtstream_out = dtrain_out[0:total_data,:]

        ftml = train_maml (ftml, meta_step, buffer_task_in, buffer_task_out,buffer_taskv_in, buffer_taskv_out)
        ftml = update_procedure(ftml,dtstream_in, dtstream_out)
        
        _, loss = model_func(ftml, dtest_in, dtest_out)
        print('Data stream Number- {} : loss = {}'.format(total_data,loss))
        if loss < threshold or j == 9:
            print("Training Finish")
            point_ftml.append(total_data)
            loss_ftml.append(loss)
            break

            

plt.figure()
plt.plot(point_ftml)
plt.xlabel('Number Of Task')
plt.ylabel('Amount of task data needed for loss threshold')
#    else:
#        for j in range (5):
#            if j==0:
#                new_task_in = train_taskx[i] 
        
#
#
#k_test1 = 60
#k_test2 = 300
#k_test3 = 600
#k_test4 = 3000
#
#fit_res1,total_loss1, ytest_test1 = eval_adaptation(trainmaml, system_rate, k_test1)
#fit_res2,total_loss2, ytest_test2 = eval_adaptation(trainmaml, system_rate, k_test2)
#fit_res3,total_loss3, ytest_test3 = eval_adaptation(trainmaml, system_rate, k_test3)
#fit_res4,total_loss4, ytest_test4 = eval_adaptation(trainmaml, system_rate, k_test4)
#
#y_out1 = fit_res1
#prediction1 = signal.savgol_filter(y_out1, 81, 3, axis=0)
#y_out2 = fit_res2
#prediction2 = signal.savgol_filter(y_out2, 81, 3, axis=0)
#y_out3 = fit_res3
#prediction3 = signal.savgol_filter(y_out3, 81, 3, axis=0)
#y_out4 = fit_res4
#prediction4 = signal.savgol_filter(y_out4, 81, 3, axis=0)
#
## Calculate Orientation Error
#euler_o1 = ytest_test1[:]
#euler_pred_ann1 = prediction1[:, :3]
#euler_ann_err1 = np.abs(euler_pred_ann1[:-anticipation_size] - euler_o1[anticipation_size:])
#
#euler_o2 = ytest_test2[:]
#euler_pred_ann2 = prediction2[:, :3]
#euler_ann_err2 = np.abs(euler_pred_ann2[:-anticipation_size] - euler_o2[anticipation_size:])
#
#euler_o3 = ytest_test3[:]
#euler_pred_ann3 = prediction3[:, :3]
#euler_ann_err3 = np.abs(euler_pred_ann3[:-anticipation_size] - euler_o3[anticipation_size:])
#
#euler_o4 = ytest_test4[:]
#euler_pred_ann4 = prediction4[:, :3]
#euler_ann_err4 = np.abs(euler_pred_ann4[:-anticipation_size] - euler_o4[anticipation_size:])
#
## Plot
#timestamp_plot = train_time_data[DELAY_SIZE:-2*anticipation_size]
#time_offset = timestamp_plot[0]
#timestamp_plot = np.array(timestamp_plot)-time_offset
#
## Split error value
##euler_ann_err_train = euler_ann_err[: int(TRAIN_SIZE*data_length)] 
##euler_ann_err_test = euler_ann_err[int((1-TEST_SIZE)*data_length):] 
#
## calculate MAE error
##ann_mae_train = np.nanmean(np.abs(euler_ann_err_train), axis=0)
#ann_mae_test1 = np.nanmean(np.abs(euler_ann_err1), axis=0)
#ann_mae_test2 = np.nanmean(np.abs(euler_ann_err2), axis=0)
#ann_mae_test3 = np.nanmean(np.abs(euler_ann_err3), axis=0)
#ann_mae_test4 = np.nanmean(np.abs(euler_ann_err4), axis=0)
#
##print('Train')
##print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_train[0], ann_mae_train[1], ann_mae_train[2]))
### print('99% ANN training, {:.2f}, {:.2f}, {:.2f}'.format(final_train_99pitch, final_train_99roll, final_train_99yaw))
#
#print('Test 60')
#print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test1[0], ann_mae_test1[1], ann_mae_test1[2]))
#print('Test 300')
#print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test2[0], ann_mae_test2[1], ann_mae_test2[2]))
#print('Test 600')
#print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test3[0], ann_mae_test3[1], ann_mae_test3[2]))
#print('Test 3000')
#print('Final Orientation MAE [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test4[0], ann_mae_test4[1], ann_mae_test4[2]))

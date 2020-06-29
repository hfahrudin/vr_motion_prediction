import tensorflow as tf
import tensorflow.keras as keras
import random
import numpy as np
from math import pi
from math import cos
from math import floor


loss_function = tf.losses.MeanSquaredError()

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def tensor_to_np(tensor):
    return (np.array(obj) for obj in tensor)


def cosine_annealing(epoch, total_epoch, lrate_max):
    cos_val = (pi*epoch)/total_epoch
    lrate_val = (lrate_max/2)*(cos(cos_val)+1)
    return lrate_val

def copy_model(model, x):
    '''
        To copy weight/variable model to the next model
    '''
    copied_model = tf.keras.models.clone_model(model)
    copied_model.forward = model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def model_func(model, x_train, t_train): #compute loss
    y_pred = model(x_train)
    loss = tf.losses.MeanAbsoluteError()(y_pred,t_train)
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
def train_maml (model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, inner_loop = 1, lr_inner = 0.001, lr_outer_max = 0.001, log_step = 100, ca = False):
    losses = []
    lr_outer = lr_outer_max
    print("Training is starting")
    for epoch in range (epochs):
        outer_loss = 0
        if ca :
            lr_outer = cosine_annealing(epoch, epochs, lr_outer_max/10)
            print(lr_outer)
        
            
        opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
        with tf.GradientTape() as outer_tape: 
            x = traintaskx[0]
            model_copy = copy_model(model,x)
            for _ in range(inner_loop):
                for i in range (len(traintaskx)): #for all dataset t with size i
                
                    x = traintaskx[i]
                    y = traintaskt[i] #convert dataset into tensor
 
                    model(x)#forward pass to initialize weights
                    #step 5
                    with tf.GradientTape() as inner_tape:
                        _,inner_loss = model_func(model, x, y)
                    #step 6
                    gradients2 = inner_tape.gradient(inner_loss, model.trainable_variables)
                    
                    k = 0
                    for j in range(len(model_copy.layers)):
                        if j % 2 == 0:
                            model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients2[k]))
                            model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients2[k+1]))
                            k += 2
            for i in range(len(valtaskx)):
                xval = valtaskx[i]
                yval = valtaskt[i]
                _, loss = model_func(model_copy, xval, yval)
                outer_loss = (outer_loss+loss)/(i+1)     

        gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(outer_loss)
        print('Step {} : loss = {}'.format(epoch,outer_loss))
    return model, losses

def orderone (model, epochs, traintaskx, traintaskt, valtaskx, valtaskt, lr_inner = 0.01, lr_outer_max = 0.01, log_step = 100, inner_loop=10):
    losses = []
    print("Training is starting")
    for epoch in range (epochs):

        outer_loss = tf.convert_to_tensor(0.0, dtype=tf.float64)
        lr_outer = 0.0001
        x_sample = traintaskx[0]
        model(x_sample)
        model_copy = copy_model(model, x_sample)
        with tf.GradientTape() as outer_tape: 
            for i in range(len(traintaskx)):
                x = traintaskx[i]
                y = traintaskt[i] #convert dataset into tensor

                for _ in range(inner_loop):
                    with tf.GradientTape() as inner_tape:
                        
                        _, inner_loss = model_func(model_copy, x, y)

                    inner_gradients = inner_tape.gradient(inner_loss, model_copy.trainable_variables)
                    keras.optimizers.SGD(learning_rate = lr_inner).apply_gradients(zip(inner_gradients, model_copy.trainable_variables))


                x_val = valtaskx[i]
                y_val = valtaskt[i]
                _, loss = model_func(model_copy, x_val, y_val)
                outer_loss+=loss

        
        outer_gradients = outer_tape.gradient(outer_loss , model_copy.trainable_variables)
        keras.optimizers.SGD(learning_rate = lr_outer).apply_gradients(zip(outer_gradients, model.trainable_variables)) 
        total_loss = outer_loss
        loss = total_loss / (len(traintaskx))
        losses.append(loss)
        print('Step{} : loss = {}'.format(epoch,loss))
    return model, losses


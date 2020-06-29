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


def cosine_annealing(epoch, total_epoch, lr_max, lr_min):
    lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * epoch / total_epoch)) / 2
    return lr

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
    loss = tf.losses.MeanAbsoluteError()(t_train, y_pred)
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
def train_maml (model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, inner_loop = 3, lr_inner = 0.001, lr_outer_max = 0.001, lr_outer_min=None, log_step = 100, ca = False, da=False):
    losses = []
    lr_outer = lr_outer_max
    if lr_outer_min == None:
        lr_outer_min = lr_outer_max/100
    print("Training is starting")
    for epoch in range (epochs):
        outer_loss = 0
        if ca :
            lr_outer = cosine_annealing(epoch, epochs, lr_outer_max, lr_outer_min)
            print(lr_outer)

        opt_outer = keras.optimizers.SGD(learning_rate=lr_outer)
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
        if epoch < (epochs//2)+1 and da:
            print("order_one")
            gradients = outer_tape.gradient(outer_loss, model_copy.trainable_variables)
        else:
            gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(outer_loss)
        print('Step {} : loss = {}'.format(epoch,outer_loss))
    return model, losses

def orderone (model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, inner_loop = 3, lr_inner = 0.001, lr_outer_max = 0.001, log_step = 100, ca = False):
    losses = []
    lr_outer = lr_outer_max
    if lr_outer_min == None:
        lr_outer_min = lr_outer_max/100
    print("Training is starting")
    for epoch in range (epochs):
        outer_loss = 0
        if ca :
            lr_outer = cosine_annealing(epoch, epochs, lr_outer_max, lr_outer_min)
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

        gradients = outer_tape.gradient(outer_loss, model_copy.trainable_variables)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(outer_loss)
        print('Step {} : loss = {}'.format(epoch,outer_loss))
    return model, losses

def importance_weights(length, multi_steps,curr):

    loss_weights = np.ones(shape=(length)) * (1.0 / length)
    decay_rate = 1.0 / length / multi_steps
    min_value_for_non_final_losses = 0.03 /length
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (curr * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (curr * (length- 1) * decay_rate),
        1.0 - ((length - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    return loss_weights

def train_maml_msl(model, epochs, traintaskx, traintaskt,valtaskx, valtaskt, inner_loop = 3, lr_inner = 0.001, lr_outer_max = 0.001, log_step = 100, ca = False, da=False, lr_outer_min = None):
    losses = []
    lr_outer = lr_outer_max
    if lr_outer_min == None:
        lr_outer_min = lr_outer_max/100
    print("Training is starting")
    for epoch in range (epochs):
        outer_loss = 0
        if ca :
            lr_outer = cosine_annealing(epoch, epochs, lr_outer_max, lr_outer_min)
            print(lr_outer)
        
        i_weights = importance_weights(inner_loop, len(valtaskx),  epoch)
        opt_outer = keras.optimizers.SGD(learning_rate=lr_outer)
        with tf.GradientTape() as outer_tape: 
            x = traintaskx[0]
            model_copy = copy_model(model,x)
            out_total = 0
            for z in range(inner_loop):
                
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
                out_total+=i_weights[z]*outer_loss    
        if epoch < (epochs//2)+1 and da:
            print("order_one")
            gradients = outer_tape.gradient(outer_loss, model_copy.trainable_variables)
        else:
            gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
        opt_outer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(out_total)
        print('Step {} : loss = {}'.format(epoch,out_total))
    return model, losses
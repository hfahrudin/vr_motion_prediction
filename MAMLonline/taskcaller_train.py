# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:55:30 2020

@author: WFS
"""
import numpy as np
import pandas as pd
from preparets import preparets
import argparse
import tensorflow as tf
from sklearn import preprocessing
from train_test_split_meta import train_test_split_meta
import matplotlib.pyplot as plt


def taskcaller_train (filename, system_rate, k, n):
    stored_df = pd.read_csv(filename)
    train_gyro_data = np.array(stored_df[['angular_vec_x', 'angular_vec_y', 'angular_vec_z']], dtype=np.float64)
    train_acce_data = np.array(stored_df[['acceleration_x', 'acceleration_y', 'acceleration_z']], dtype=np.float64)
    train_eule_data = np.array(stored_df[['input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw']], dtype=np.float64)
    train_time_data = np.array(stored_df['timestamp'], dtype=np.int64)
    train_time_data = train_time_data / 705600000
    train_data_id = stored_df.shape[0]
    print(filename +' loaded...\n')
    '''
    #####   TRAINING DATA PREPROCESSING    #####
    '''
    parser = argparse.ArgumentParser(description='Offline Motion Prediction')
    parser.add_argument('-a', '--anticipation', default=300, type=int)
    args = parser.parse_args()
    
    # Remove zero data from collected training data
    idle_period = int(2 * system_rate)
    train_gyro_data = train_gyro_data* 180/ np.pi
    train_gyro_data = train_gyro_data[idle_period:train_data_id, :]
    train_acce_data = train_acce_data[idle_period:train_data_id, :]
    train_eule_data = train_eule_data[idle_period:train_data_id, :]
    train_time_data = train_time_data[idle_period:train_data_id]
    train_eule_data = train_eule_data * 180 / np.pi
    train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
    train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float64), train_alfa_data])
    train_alfa_data = train_alfa_data * np.pi / 180
    train_gyro_data = train_gyro_data * np.pi / 180
    train_acce_data = train_acce_data / 9.8
#    
#    ori_mean = np.nanmean(np.abs(train_acce_data), axis=0)
#    ori_std = np.std(train_acce_data, axis=0)
#    plt.figure()
#    plt.scatter(train_gyro_data[:,0], train_gyro_data[:,2])
#    plt.xlim(0, 1)
#    plt.ylim(0, 1)
#    plt.title('Angular Velocity Scatter Plot')
#    plt.xlabel('X (Degree/Second)')
#    plt.ylabel('Z (Degree/Second)')
#    plt.show()
#    
#    print('Mean dataset pitch', np.mean(ori_mean[0]))
#    print('Mean dataset roll', np.mean(ori_mean[1]))
#    print('Mean dataset yaw', np.mean(ori_mean[2]))
#    
#    print('std dataset pitch', np.mean(ori_std[0]))
#    print('std dataset roll', np.mean(ori_std[1]))
#    print('std dataset yaw', np.mean(ori_std[2]))
    # Create data frame of all features and smoothing
    sliding_window_time = 100
    sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))
    
    ann_feature = np.column_stack([train_eule_data, 
                                   train_gyro_data, 
                                   train_acce_data, 
    #                               train_magn_data,
                                   ])
        
    feature_name = ['pitch', 'roll', 'yaw', 
                    'gX', 'gY', 'gZ', 
                    'aX', 'aY', 'aZ', 
    #                'mX', 'mY', 'mZ', 
    #                'EMG1', 'EMG2', 'EMG3', 'EMG4',
                    ]
    
    ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
    ann_feature_df = ann_feature_df.rolling(sliding_window_size, center=True, min_periods=1).mean()
    
    
    # Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
    anticipation_time = args.anticipation
    anticipation_size = int(np.round(anticipation_time * system_rate / 1000))
    
    spv_name = ['pitch', 'roll', 'yaw']
    target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)
    input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)
    
    input_nm = len(input_series_df.columns)
    target_nm = len(target_series_df.columns)
    
    
    '''
    #####   NEURAL NETWORK MODEL TRAINING    #####
    '''
    
    
    # Neural network parameters
    DELAY_SIZE = int(25 * (system_rate / 250))
    
    # Import datasets
    input_series = np.array(input_series_df)
    target_series = np.array(target_series_df)
    
    
    # Normalized input and target series (updated)
    #input_norm = preprocessing.scale(input_series)
    normalizer = preprocessing.StandardScaler()
    normalizer.fit(input_series)
    input_norm = normalizer.transform(input_series)
    
    
    # Reformat the input into TDNN format
    x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
    data_length = x_seq.shape[0]
    print('Anticipation time: {}ms\n'.format(anticipation_time))
    
    xtask_train, ytask_train,xtask_val, ytask_val, dtrainx, dtrainy, dtestx, dtesty = train_test_split_meta(x_seq, t_seq, k, n)
    
    xtask_train = tf.convert_to_tensor(xtask_train)
    ytask_train = tf.convert_to_tensor(ytask_train)
    xtask_val = tf.convert_to_tensor(xtask_val)
    ytask_val = tf.convert_to_tensor(ytask_val)
    
    dtrainx = tf.convert_to_tensor(dtrainx)
    dtrainy = tf.convert_to_tensor(dtrainy)
    dtestx = tf.convert_to_tensor(dtestx)
    dtesty = tf.convert_to_tensor(dtesty)
    
    return  xtask_train, ytask_train,xtask_val,ytask_val, dtrainx, dtrainy, dtestx, dtesty, input_nm, target_nm, data_length, DELAY_SIZE, train_eule_data, anticipation_size, train_time_data

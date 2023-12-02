#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:28:06 2022

build model for cubloc method

@author: zhangj2
"""
# In[]
import keras.backend as K

import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,Conv3D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape,Conv3DTranspose
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D,Softmax 

from keras.layers import UpSampling3D,MaxPooling3D

import tensorflow as tf


 
def up_and_concate_3d(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[4]
    out_channel = in_channel // 2
    up = Conv3DTranspose(out_channel, [2, 2, 2], strides=[2, 2, 2], padding='valid')(down_layer)
    
    print("--------------")
    print(str(up.get_shape()))
 
    print(str(layer.get_shape()))
    print("--------------")
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=4))
 

    net1_shape = layer.get_shape().as_list()
    
    offsets = [0,0, 0, 0 , 0]
    
    if not net1_shape[1] is None:
        size = [-1, net1_shape[1], net1_shape[1],net1_shape[3], -1]
        up1 = tf.slice(up, offsets, size)   
          
    concate = my_concat([up1, layer])
    # must use lambda
    # concate=K.concatenate([up, layer], 3)
    return concate

def up_and_concate_wu(down_layer, layer,con=True):
    in_channel = down_layer.get_shape().as_list()[4]
    out_channel = in_channel // 2
    up = Conv3DTranspose(out_channel, [2, 2, 2], strides=[2, 2, 2], padding='valid')(down_layer)
    
    print("--------------")
    print(str(up.get_shape()))
 
    print(str(layer.get_shape()))
    print("--------------")
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=4))
 

    net1_shape = layer.get_shape().as_list()
    
    offsets = [0,0, 0, 0 , 0]
    
    if not net1_shape[1] is None:
        size = [-1, net1_shape[1], net1_shape[1],net1_shape[3], -1]
        up1 = tf.slice(up, offsets, size)   
    if con:
        concate = my_concat([up1, layer])
    else:
        concate =up1
    # must use lambda
    # concate=K.concatenate([up, layer], 3)
    return concate


def up_and_concate3(down_layer, layer1,layer2):
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=4))
    in_channel = down_layer.get_shape().as_list()[4]
    out_channel = in_channel // 2
    up = Conv3DTranspose(out_channel, [2, 2, 2], strides=[2, 2, 2], padding='valid')(down_layer)
    layer = my_concat([layer1, layer2])
    print("--------------")
    print(str(up.get_shape()))
 
    print(str(layer.get_shape()))
    print("--------------")
      
    net1_shape = layer.get_shape().as_list()
    
    offsets = [0, 0, 0, 0 , 0]
    
    if not net1_shape[1] is None:
        size = [-1, net1_shape[1], net1_shape[1],net1_shape[3], -1]
        up1 = tf.slice(up, offsets, size)   
          
    concate = my_concat([up1, layer])
    # must use lambda
    # concate=K.concatenate([up, layer], 3)
    return concate 
 
def attention_block_3d(x, g, inter_channel):
    '''
    :param x: x input from down_sampling same layer output x(?,x_height,x_width,x_depth,x_channel)
    :param g: gate input from up_sampling layer last output g(?,g_height,g_width,g_depth,g_channel)
    g_height,g_width,g_depth=x_height/2,x_width/2,x_depth/2
    :return:
    '''
    # theta_x(?,g_height,g_width,g_depth,inter_channel)
    theta_x = Conv3D(inter_channel, [2, 2, 2], strides=[2, 2, 2])(x)
 
    # phi_g(?,g_height,g_width,g_depth,inter_channel)
    phi_g = Conv3D(inter_channel, [1, 1, 1], strides=[1, 1, 1])(g)
 
    # f(?,g_height,g_width,g_depth,inter_channel)
    f = Activation('relu')(keras.layers.add([theta_x, phi_g]))
 
    # psi_f(?,g_height,g_width,g_depth,1)
    psi_f = Conv3D(1, [1, 1, 1], strides=[1, 1, 1])(f)
 
    # sigm_psi_f(?,g_height,g_width,g_depth)
    sigm_psi_f = Activation('sigmoid')(psi_f)
 
    # rate(?,x_height,x_width,x_depth)
    rate = UpSampling3D(size=[2, 2, 2])(sigm_psi_f)
 
    # att_x(?,x_height,x_width,x_depth,x_channel)
    att_x = keras.layers.multiply([x, rate])
 
    return att_x
 
 
def unet_model_3d(input_shape, n_labels, batch_normalization=False, out=100,initial_learning_rate=0.00001):
    """
    input_shape:without batch_size,(img_height,img_width,img_depth)
    metrics:
    """
 
    inputs = Input(input_shape,name='input')
 
    down_layer = []
 
    layer = inputs
    depths=3
    
    for depth in range(depths):
        fliters=4*2**depth
        layer = res_block_v2_3d(layer, fliters, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2],padding='same')(layer)
        # print(str(layer.get_shape()))
    
    # bottle_layer
    layer = res_block_v2_3d(layer, fliters*2, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))
 
    for depth in range(depths-1,-1,-1):    
        layer = up_and_concate_3d(layer, down_layer[depth])
        layer = res_block_v2_3d(layer, 4*2**depth, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
 

    # score_layer
    layer = Conv3D(n_labels, [1, 1, 1], strides=[1, 1, 1])(layer)
    print(str(layer.get_shape()))
    lshape=layer.get_shape().as_list()
    
    layer = Reshape((lshape[1],lshape[2],lshape[3]))(layer)
    layer = Conv2D(out, 1)(layer)
 
    # softmax
    outputs = Activation('relu',name='output')(layer)
    print(str(outputs.get_shape()))

 
    model = Model(inputs=inputs, outputs=outputs)
 
    # model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),metrics=['accuracy'])
 
    return model
 
 
def res_block_v2_3d(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                    padding='same'):
    input_n_filters = input_layer.get_shape().as_list()[3]
    print(str(input_layer.get_shape()))
    layer = input_layer

    for i in range(2):
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv3D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
 
    if out_n_filters != input_n_filters:
        skip_layer = Conv3D(out_n_filters, [1, 1, 1], strides=stride, padding=padding)(input_layer)
    else:
        skip_layer = input_layer
 
    out_layer = keras.layers.add([layer, skip_layer])
 
    return out_layer


# In[]
def wu_model_3d(input_shape, n_labels, batch_normalization=False, out=100,initial_learning_rate=0.00001,con=True,fl=False):
    """
    input_shape:without batch_size,(img_height,img_width,img_depth)
    metrics:
    """
 
    inputs = Input(input_shape,name='input')
 
    down_layer = []
 
    layer = inputs
    depths=3
    
    for depth in range(depths):
        fliters=4*2**depth
        layer = res_block_v2_3d(layer, fliters, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2],padding='same')(layer)
        # print(str(layer.get_shape()))
    
    # bottle_layer
    layer = res_block_v2_3d(layer, fliters*2, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))
 
    for depth in range(depths-1,-1,-1):  
        if fl:
            if depth==0:
                layer = up_and_concate_wu(layer, down_layer[depth],con=False)
            else:
                layer = up_and_concate_wu(layer, down_layer[depth])
            
        else:
            layer = up_and_concate_wu(layer, down_layer[depth],con=con)
        
        layer = res_block_v2_3d(layer, 4*2**depth, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
 

    # score_layer
    layer = Conv3D(n_labels, [1, 1, 1], strides=[1, 1, 1])(layer)
    print(str(layer.get_shape()))
    lshape=layer.get_shape().as_list()
    
    layer = Reshape((lshape[1],lshape[2],lshape[3]))(layer)
    layer = Conv2D(out, 1)(layer)
 
    # softmax
    outputs = Activation('relu',name='output')(layer)
    print(str(outputs.get_shape()))

 
    model = Model(inputs=inputs, outputs=outputs)
 
    # model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),metrics=['accuracy'])
 
    return model
 
# In[]






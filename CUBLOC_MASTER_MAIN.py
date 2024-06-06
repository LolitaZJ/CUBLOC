#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:33:34 2023

CUBLOC:
Multi-station seismic location based on 3D U-Net

#============#
python CUBLOC_MASTER_MAIN.py --mode=train

python CUBLOC_MASTER_MAIN.py --mode=test
#============#

@author: zhangj2
"""
#=============================================================#
import os
os.getcwd()
import tensorflow as tf
#=============================================================#
import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,Conv3D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D,Softmax
from keras.regularizers import l1
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras import backend as K
from tensorflow.keras.utils import plot_model
#=============================================================#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
import datetime
from obspy.signal.trigger import recursive_sta_lta,classic_sta_lta,trigger_onset
from scipy import signal
from obspy.core import UTCDateTime
import pandas as pd
import h5py
import argparse
#=============================================================#
from CUBLOC_model import unet_model_3d,wu_model_3d
from CUBLOC_utils import plot_3d,plt_3d_wave,plot_loss,DataGenerator_CULOC_New,eva_model,plt_2d_wave

# In[] gpu and configures
# Set GPU
#==========================================# 
def start_gpu(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('Physical GPU：', len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('Logical GPU：', len(logical_gpus))

#==========================================#
# Set Configures
#==========================================# 
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="3",
                        help="set gpu ids") 
    
    parser.add_argument("--input_size",
                        default=(50,50,1500,3),
                        help="input size (x,y,t,c)")    

    parser.add_argument("--dr",
                        default=2,
                        type=int,
                        help="Gaussain Radius")

    parser.add_argument("--dx",
                        default=20,
                        type=int,
                        help="Size of grid 111/dx")

    parser.add_argument("--grid",
                        default=[50,50],
                        type=list,
                        help="Size of grid 111/dx")

    parser.add_argument("--s_range",
                        default=[-99.0,-96.5,35.0,37.5],
                        type=list,
                        help="Focus region")

    parser.add_argument("--model_name",
                        default="CUBLOC_M01",
                        help="model name")
    
    parser.add_argument("--save_path",
                        default="./GIT_LOC",
                        help="save result path") 
    
    parser.add_argument("--save_fig",
                        default="figure",
                        help="save result path")

    parser.add_argument("--data_path",
                        default="/your/hdf5/data/path/data.hdf5",
                        help="data path")   

    parser.add_argument("--csv_path",
                        default="/your/csv/data/path/data.csv",
                        help="csv path") 
    
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs (default: 100)")
    
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="early stopping")

    parser.add_argument("--workers",
                        default=2,
                        type=int,
                        help="workers")
    
    parser.add_argument("--multiprocessing",
                        default=False,
                        help="use_multiprocessing")  
  
    parser.add_argument("--monitor",
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='mse',
                        help="loss fucntion")  
    
    parser.add_argument("--mode",
                        default='test',
                        help="train/test/")         
    
    args = parser.parse_args()
    return args



# In[] main
if __name__ == '__main__':
    # In[]
    
    args = read_args()
    # set gpu
    start_gpu(args)

    #=====================select data============================#
    df=pd.read_csv(args.csv_path) # load event info
    fsc=h5py.File(args.data_path,'r') # load dataset   
    si_names=df['source_id'].tolist()
    si_unique=list(set(si_names))
    si_inx=[]
    for i in si_unique:
        si_inx.append(si_names.count(i))
    si_id=[i for i in range(len(si_inx)) if si_inx[i]>3 ]
    #=============================================================#
    # set range
    x0 = args.s_range[0]
    y0 = args.s_range[2]
    x1 = args.s_range[1]
    y1 = args.s_range[3]
    #print(x0,x1,y0,y1)
    #=============================================================#
    # get suitable events
    si_id_new=[]
    for i in range(len(si_id)):
        si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0]
        si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]
        if si_lon>x0 and si_lon<x1 and si_lat>y0 and si_lat<y1:
            si_id_new.append(si_id[i])
    si_id = si_id_new
    #=============================================================# 

    # ===========================mk generator=====================#
    batch_size=args.batch_size
    dr=args.dr
    dx=args.dx
    w1=args.grid[0]
    w2=args.grid[1]
    sinx=int(len(si_id)*0.8)
    #print(sinx,si_id[0],si_unique[0])
    gen_train=DataGenerator_CULOC_New(df,fsc,si_id[:sinx],si_unique,batch_size=batch_size,dr=dr,x0=x0,y0=y0,dx=dx,w1=w1,w2=w2 )
    
    gen_valid=DataGenerator_CULOC_New(df,fsc,si_id[sinx:],si_unique,batch_size=batch_size,dr=dr,x0=x0,y0=y0,dx=dx,w1=w1,w2=w2 )
    
    val_len=(len(si_id)-sinx)
    validation_steps=val_len//batch_size
    steps_per_epoch=sinx//batch_size
    epochs=args.epochs
    model_name=args.model_name+'_r%d'%dr
    monitor=args.monitor
    mode=args.monitor_mode
    patience=args.patience
    use_multiprocessing=args.multiprocessing
    workers=args.workers
    save_path=args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    
    saveBestModel= ModelCheckpoint('%s/%s.h5'%(save_path,model_name), monitor=monitor, verbose=1, save_best_only=True,mode=mode)
    estop = EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode=mode)
    callbacks_list = [saveBestModel,estop]
    if args.mode=='train':
        # ========================= bulid model======================#
        wave_input=args.input_size
        # 3D u-net
        # model=unet_model_3d(wave_input,1,batch_normalization=False) 
        # our model
        model=wu_model_3d(wave_input, 1,batch_normalization=False,fl=True)
        # model.summary()
    
        # ========================= model fit======================#
        model.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        begin = datetime.datetime.now() 
        history_callback=model.fit_generator(
                                          generator=gen_train, 
                                          steps_per_epoch= steps_per_epoch,                      
                                          epochs=epochs, 
                                          verbose=1,
                                          callbacks=callbacks_list,
                                          use_multiprocessing=use_multiprocessing,
                                          workers=workers,
        #                                     validation_split=0.1)
                                 validation_data=gen_valid,
                                 validation_steps=validation_steps)    
                            
        end = datetime.datetime.now()
        print(end-begin)
        
        plot_loss(history_callback,save_path=save_path+'/',model=model_name,c='c')
        model.save_weights('%s/%s.wt'%(save_path,model_name))
    
    if args.mode=='test':
    
        # ========================= load model ========================= #
        model_name=args.model_name+'_r%d'%dr
        model=load_model('%s/%s.h5'%(save_path,model_name))
        

        # ========================= load and predict dataset ========================= #
        gen=DataGenerator_CULOC_New(df,fsc,si_id[sinx:],si_unique,batch_size=val_len,dr=dr,x0=x0,y0=y0,dx=dx,w1=w1,w2=w2 )
        
        tmp = iter(gen)
        tmp1 = next(tmp)
        wave_inp = tmp1[0]['input']
        wave_oup = tmp1[1]['output']
        #print(np.max( np.max(wave_inp[0,:,:,:,2]) ) )
        # ========================= predict ========================= #
        begin = datetime.datetime.now() 
        pred_test=model.predict(wave_inp[:,:,:,:,:])
        end = datetime.datetime.now() 
        print(end-begin)

        # ========================= calculate error ========================= #
        pred_loc=[]
        ca_loc=[]
        
        for i in range(0,val_len,1):
            # print(np.max(wave_oup[i,:,:,:]))
            # print(np.max(pred_test[i,:,:,:]))
            res2=eva_model(pred_test[i,:,:,:],r=0.5,n=9)
            res1=eva_model(wave_oup[i,:,:,:],r=0.5,n=9)
            ca_loc.append([y0+res1[0][0]*5.5/111,x0+res1[0][1]*5.5/111,res1[0][2] ])
            pred_loc.append([y0+res2[0][0]*5.5/111,x0+res2[0][1]*5.5/111,res2[0][2] ])
        
        pred_loc=np.array(pred_loc)
        ca_loc=np.array(ca_loc)

        # ========================= plt error hist ========================= #
    
        error_loc=ca_loc-pred_loc
        
        error_loc[:,0]=error_loc[:,0]*111
        error_loc[:,1]=error_loc[:,1]*111
        
        ord3=np.linalg.norm(error_loc,axis=1)
        print( np.mean(ord3) )

        #===================================================================#
        save_fig='%s/%s'%(save_path,args.save_fig)
        if not os.path.exists(save_fig):
            os.makedirs(save_fig)

        #===================================================================#
        font2 = {'family' : 'Times New Roman','weight' : 'bold','size' : 20,}
        figure, ax = plt.subplots(figsize=(10,8))
        res_err=[i for i in ord3 if i <30]
        plt.hist(res_err,edgecolor='k')
        plt.tick_params(labelsize=20)
        plt.xlabel('Error (km)',font2)
        plt.ylabel('Count',font2)
        plt.grid(ls='-.')
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.savefig('%s/New_hist_r%d'%(save_fig,dr),dpi=600)
        plt.show()

        # ============================ simply QC ==========================#
        res_plt_id=[i for i in range(len(ord3)) if ord3[i]<20]    
        for i in res_plt_id[:3]:    
            path='%s/Res_%s_%d_3d'%(save_fig,model_name,i)
            evt=eva_model(wave_oup[i,:,:,:],r=0.5,n=9)
            plt_3d_wave(wave_inp,evt,i,path=path)
            
            path='%s/Res_%s_%d_d'%(save_fig,model_name,i)
            plot_3d(abs(wave_inp[i,:,:,:,0]),name='Data',path=path)
            
            path='%s/Res_%s_%d_l'%(save_fig,model_name,i)
            plot_3d(abs(wave_oup[i,:,:,:50]),name='Labeled',fl=1,path=path)
            
            path='%s/Res_%s_%d_p'%(save_fig,model_name,i)
            plot_3d(abs(pred_test[i,:,:,:50]),name='Predicted',fl=1,path=path)
            
            path='%s/Res_%s_%d_2d'%(save_fig,model_name,i)
            plt_2d_wave(wave_inp,i,path)

# In[]














#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:38:11 2022

utils for cubloc method

@author: zhangj2
"""
# In[] mk data
import numpy as np
import h5py
import matplotlib.pyplot as plt
try:
    from keras.utils import Sequence
except:
    from tensorflow.keras.utils import Sequence
import math    
from scipy import signal
from skimage import measure
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
from obspy.core import UTCDateTime
import math

# In[]
def wave_num(wave_inp,i):
    n=0
    for ii in range(50):
        for jj in range(50):
            if np.sum(abs(wave_inp[i,ii,jj,:,0]))>0:
                n=n+1
    return n

# In[]
def plt_2d_wave_ps(wave_inp,wave_inp1,i,path):
    n=0
    font2 = {'family' : 'Times New Roman',    'weight' : 'normal',    'size'   : 18,    }
    figure, ax = plt.subplots(figsize=(8,12))
    for ii in range(50):
        for jj in range(50):
            if np.sum(abs(wave_inp1[i,ii,jj,:,0]))>0:
                plt.plot(n+wave_inp[i,ii,jj,:,0]/ np.max(abs(wave_inp[i,ii,jj,:,0])),'k')
                plt.plot(n+wave_inp1[i,ii,jj,:,0]/ np.max(abs(wave_inp1[i,ii,jj,:,0])),'r')
                n=n+1
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Samples',font2)
    plt.ylabel('Stations',font2)
    plt.xlim([0,1500])
    plt.ylim([-1,n])
    if path is not None:
        plt.savefig(path,dpi=600)     
    plt.show()

# In[]
def plt_2d_wave(wave_inp,i,path):
    n=0
    font2 = {'family' : 'Times New Roman',    'weight' : 'normal',    'size'   : 18,    }
    figure, ax = plt.subplots(figsize=(8,12))
    for ii in range(50):
        for jj in range(50):
            if np.sum(abs(wave_inp[i,ii,jj,:,0]))>0:
                plt.plot(n+wave_inp[i,ii,jj,:,0]/ np.max(abs(wave_inp[i,ii,jj,:,0])),'k')
                n=n+1
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Samples',font2)
    plt.ylabel('Stations',font2)
    plt.xlim([0,1500])
    plt.ylim([-1,n])
    if path is not None:
        plt.savefig(path,dpi=600)     
    plt.show()
    
# In[] location association
def gaussian(x, sigma,  u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return y/np.max(abs(y))

# In[]
class DataGenerator_CULOC_New(Sequence):

    def __init__(self,df,fsc,si_id,si_unique,batch_size=128,w1=50,w2=50,x0=-99,y0=35,dx=20,
                 ave_single=False,shift=False,dp=0.1,tt=2,mul=False,tri=True,dr=4,
                 shuffle=True,rot=False,rot90=False,evt_fg=False):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
            
        self.batch_size = batch_size
        self.df=df
        self.fsc=fsc
        self.si_id=si_id
        self.si_unique=si_unique
        self.indexes=np.arange(len(self.si_id))
        self.shuffle = shuffle
        self.w1=w1
        self.w2=w2
        self.x0=x0
        self.y0=y0
        self.dx=dx
        self.rot=rot
        self.rot90=rot90
        self.shift=shift
        self.ave_single=ave_single
        self.dp=dp
        self.tt=tt
        self.mul=mul
        self.tri=tri
        self.dr=dr
        self.evt_fg=evt_fg
        

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.si_id)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        # read batch data
        if self.evt_fg:
            X, Y, Z= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
            return ({'input': X[:,:self.w1,:self.w2,:3000,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] },{'evt_id':Z})             
        else:
            X, Y= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
            return ({'input': X[:,:self.w1,:self.w2,:3000,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def _mk_lab(self,x,y,z,d=10):
        v=np.zeros((self.w1,self.w2,100))
        for i in range(self.w1):
            for j in range(self.w2):
                for k in range(100):
                    v[i,j,k]=np.exp(-((x-i*1.0)**2+(y-j*1.0)**2+(z-k*1.0)**2)/2/d**2)
                    #/(2*math.pi)**(3/2)/d**3
        return v
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1 
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
        
    def _rotate2(self,x,y,an,x1,y1):
        
        halfx = int( np.floor(self.w1 / 2) )
        halfy = int( np.floor(self.w2 / 2) )  
        x=[i-halfx for i in x]
        y=[i-halfy for i in y]
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y+halfx
        new_y=np.cos(an)*y+np.sin(an)*x+halfy
        # b_x1,b_x2 = np.min(new_x),np.max(new_x)
        # b_y1,b_y2 = np.min(new_y),np.max(new_y)
        x1=x1-halfx
        y1=y1-halfy
        s_x=np.cos(an)*x1-np.sin(an)*y1+halfx
        s_y=np.cos(an)*y1+np.sin(an)*x1 +halfy 
        
        new_x=[int(np.round(i)) for i in new_x]
        new_y=[int(np.round(i)) for i in new_y]
        # return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1    
        return new_x,new_y,int(s_x),int(s_y)   
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        for i in range(data.shape[0]):
            data1=data[i,:,:,:,:]
            x_max=np.max(np.max(abs(data1)))
            if x_max!=0.0:
                data2[i,:,:,:,:]=data1/x_max 
        return data2   
    
    def _read_data(self, batch_inds,si_id,si_unique,df,fsc):
        ll=len(batch_inds)
        nn=1
        if self.rot:
            ll+=len(batch_inds)
            nn+=1
        if self.rot90:
            ll+=len(batch_inds)*3
            nn+=1
        if self.shift:
            ll+=len(batch_inds) 
            nn+=1
        if self.mul:
            ll+=len(batch_inds) 
            

        wave=np.zeros((ll,self.w1,self.w2,1500,3))
        label=np.zeros((ll,self.w1,self.w2,100,1))      
        evt_ids=[]
     
        ii=0     
        for i in batch_inds: 
            si_trace=df[(df.source_id==si_unique[si_id[i]])]['trace_name'].tolist()  
            data=[]
            for name in si_trace:
                temp=np.array(fsc.get('data/'+name))
                for iii in range(3):
                    temp[iii,:]=self._taper(temp[iii,:],1,100) 
                temp = self._bp_filter(temp,2,1,45,0.01)
                if len(temp[:,:36000][1])<36000:
                    tmp1=np.zeros((3,36000))
                    tmp1[:,:len(temp[:,:][1])]=temp[:,:]
                    data.append(tmp1)
                else:
                    data.append(temp[:,:36000]) 
            try:
                data=np.array(data) 
            except:
                print(si_unique[si_id[i]])
            #print(np.max(np.max(data)))
            sta_lon=df[(df.source_id==si_unique[si_id[i]])]['receiver_longitude'].tolist() 
            sta_lat=df[(df.source_id==si_unique[si_id[i]])]['receiver_latitude'].tolist() 
            sta_tp=df[(df.source_id==si_unique[si_id[i]])]['p_arrival_sample'].tolist()
            
            si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0] 
            si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]         
            si_dep=df[(df.source_id==si_unique[si_id[i]])]['source_depth_km'].tolist()[0] 
            gsi_dep=int(si_dep/1000)

            x0= self.x0
            y0= self.y0
            dx= self.dx
            gd_lon=[int( np.round((i1-x0)*dx)) for i1 in sta_lon ]
            gd_lat=[int( np.round((i1-y0)*dx)) for i1 in sta_lat ]
            
            gsi_lon=int(np.round((si_lon-x0)*dx))
            gsi_lat=int(np.round((si_lat-y0)*dx)) 
            # print(x0,y0,si_lon,si_lat)
            # print(gsi_lon,gsi_lat)
            tmp=[]
            for k in range(len(gd_lon)):
                # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                if self.ave_single:
                    try:
                        mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    except:
                        print(sta_tp[k])
                        mm=0
                else:
                    mm=0
                if mm!=0:
                    tmp=data[k,:,6000:9000:2]/mm
                else:
                    tmp=data[k,:,6000:9000:2]
                if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0 and gd_lat[k]<self.w2:
                    if self.tri:
                        if sta_tp[k]<9000:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                    else:
                        wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)

            label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
            ii+=1
            
            if self.evt_fg: evt_ids.append(si_unique[si_id[i]])
            
            np.random.seed(ii)
            #=========================#
            if self.shift: 
                
                tmp=[]
                drop=False
                cn=len(gd_lon)
                rg=np.arange(cn)
                if cn>10:
                    drop=True  
                for k in rg[:cn]:
                    if drop:
                        if np.random.rand()<self.dp:
                            continue
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]))
                    if self.tt==0:
                        sft=0  
                    else:
                        sft=int(np.random.uniform(-self.tt,self.tt,1)*100)
                    
                    sst=6000+sft
                    sed=sst+3000
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,sst:sed:2]/mm
                    else:
                        tmp=data[k,:,sst:sed:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        if self.tri:
                            if sta_tp[k]<9000+sft:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0) 
                            
                            
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1              
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])
            #=========================#
            if self.rot90:
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-1,::-1,:,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-1,::-1,:,:,0],axes=(1,0,2) )  
                ii+=1  
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-2,:,::-1,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-2,:,::-1,:,0],axes=(1,0,2) )  
                ii+=1   
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])
                wave[ii,:,:,:,:] = np.transpose(wave[ii-3,::-1,::-1,:,:],axes=(0,1,2,3) )             
                label[ii,:,:,:,0] =  np.transpose(label[ii-3,::-1,::-1,:,0],axes=(0,1,2) ) 
                ii+=1                  
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])
            #=========================#
            if self.rot:            

                an=np.pi/np.random.randint(4,10)*np.random.choice([-1,1])              
                gd_lon,gd_lat,gsi_lon,gsi_lat=self._rotate2(gd_lon,gd_lat,an,gsi_lon,gsi_lat)
                    
                tmp=[]
                for k in range(len(gd_lon)):
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        # wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        if self.tri:
                            if sta_tp[k]<9000:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)                        
                        
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1  
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])

        if self.mul:
            
            for i in range(len(batch_inds)):
                n1=np.random.randint(2,4)
                ex=[]
                for i1 in range(n1):
                    i2=np.random.randint(0,len(batch_inds))
                    
                    if i2 in ex:
                        if nn>4:
                            wave[ii,:,:,:,:]+=wave[i2*nn+4,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn+4,:,:,:,0]   
                        else:
                            i2=i2-1
                            ex.append(i2)
                            wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]                            
                    else:
                        ex.append(i2)
                        wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                        label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]
                        
                ii+=1
                if self.evt_fg: evt_ids.append(si_unique[si_id[i]])

        # return wave,label  
        if self.evt_fg:
            if self.ave_single:
                return wave,label,evt_ids  
            else:
                return self._normal3(wave),label,evt_ids 
        else: 
            if self.ave_single:
                return wave,label  
            else:
                return self._normal3(wave),label    
        
# In[]
def cal_db(s,n):
    sl=len(s)
    nl=len(n)
    s2=np.sum(s**2)/sl
    n2=np.sum(n**2)/nl
    snr=10*math.log10(s2/n2)
    return snr



# In[]
class DataGenerator_SCSN(Sequence):

    def __init__(self,df,fsc,si_id,si_unique,batch_size=128,w1=50,w2=50,x0=-119,y0=32.5,dx=20,
                 ave_single=False,shift=False,dp=0.1,tt=2,mul=False,tri=False,dr=2,
                 shuffle=True,rot=False,rot90=False):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
            
        self.batch_size = batch_size
        self.df=df
        self.fsc=fsc
        self.si_id=si_id
        self.si_unique=si_unique
        self.indexes=np.arange(len(self.si_id))
        self.shuffle = shuffle
        self.w1=w1
        self.w2=w2
        self.x0=x0
        self.y0=y0
        self.dx=dx
        self.rot=rot
        self.rot90=rot90
        self.shift=shift
        self.ave_single=ave_single
        self.dp=dp
        self.tt=tt
        self.mul=mul
        self.tri=tri
        self.dr=dr

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.si_id)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        # read batch data
        X, Y= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
        return ({'input': X[:,:self.w1,:self.w2,:3000,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def _mk_lab(self,x,y,z,d=10):
        v=np.zeros((self.w1,self.w2,100))
        for i in range(self.w1):
            for j in range(self.w2):
                for k in range(100):
                    v[i,j,k]=np.exp(-((x-i*1.0)**2+(y-j*1.0)**2+(z-k*1.0)**2)/2/d**2)
                    #/(2*math.pi)**(3/2)/d**3
        return v
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1 
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
        
    def _rotate2(self,x,y,an,x1,y1):
        
        halfx = int( np.floor(self.w1 / 2) )
        halfy = int( np.floor(self.w2 / 2) )  
        x=[i-halfx for i in x]
        y=[i-halfy for i in y]
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y+halfx
        new_y=np.cos(an)*y+np.sin(an)*x+halfy
        # b_x1,b_x2 = np.min(new_x),np.max(new_x)
        # b_y1,b_y2 = np.min(new_y),np.max(new_y)
        x1=x1-halfx
        y1=y1-halfy
        s_x=np.cos(an)*x1-np.sin(an)*y1+halfx
        s_y=np.cos(an)*y1+np.sin(an)*x1 +halfy 
        
        new_x=[int( np.round(i)) for i in new_x]
        new_y=[int( np.round(i)) for i in new_y]
        # return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1    
        return new_x,new_y,int(np.round(s_x)),int(np.round(s_y))   
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        for i in range(data.shape[0]):
            data1=data[i,:,:,:,:]
            x_max=np.max(np.max(abs(data1)))
            if x_max!=0.0:
                data2[i,:,:,:,:]=data1/x_max 
        return data2   
    
    def _read_data(self, batch_inds,si_id,si_unique,df,fsc):
        ll=len(batch_inds)
        nn=1
        if self.rot:
            ll+=len(batch_inds)
            nn+=1
        if self.rot90:
            ll+=len(batch_inds)*3
            nn+=1
        if self.shift:
            ll+=len(batch_inds) 
            nn+=1
        if self.mul:
            ll+=len(batch_inds) 
        rm_inx=['CRY','KNW','FRD','RRSP',
                'SND','B082','B086','B087',
                'B093','BZN','B088','JORD',
                'RHIL','TFRD','GVAR1','TMSP','B081']            

        wave=np.zeros((ll,self.w1,self.w2,1500,3))
        label=np.zeros((ll,self.w1,self.w2,100,1))            
     
        ii=0     
        for i in batch_inds: 
            si_trace=df[(df.source_id==si_unique[si_id[i]])]['trace_name'].tolist()  
            data=[]
            for name in si_trace:
                temp=np.array(fsc.get('data/'+name))
                for iii in range(3):
                    temp[iii,:]=self._taper(temp[iii,:],1,100) 
                temp = self._bp_filter(temp,2,1,45,0.01)
                if len(temp[:,:36000][1])!=36000:
                    tmp1=np.zeros((3,36000))
                    tmp1[:,:len(temp[:,:36000][1])]=temp
                    data.append(tmp1)
                else:
                    data.append(temp[:,:36000]) 
            try:
                data=np.array(data) 
            except:
                print(si_unique[si_id[i]])
            
            sta_lon=df[(df.source_id==si_unique[si_id[i]])]['receiver_longitude'].tolist() 
            sta_lat=df[(df.source_id==si_unique[si_id[i]])]['receiver_latitude'].tolist() 
            sta_nas=df[(df.source_id==si_unique[si_id[i]])]['receiver_code'].tolist() 
            
            sta_tp=df[(df.source_id==si_unique[si_id[i]])]['p_arrival_sample'].tolist()
            
            si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0] 
            si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]         
            si_dep=df[(df.source_id==si_unique[si_id[i]])]['source_depth_km'].tolist()[0] 
            gsi_dep=int(np.round(si_dep/1000))

            x0= self.x0
            y0= self.y0
            dx= self.dx
            gd_lon=[int( np.round((i1-x0)*dx)) for i1 in sta_lon ]
            gd_lat=[int( np.round((i1-y0)*dx)) for i1 in sta_lat ]
            
            gsi_lon=int( np.round((si_lon-x0)*dx))
            gsi_lat=int( np.round((si_lat-y0)*dx))  
            
            tmp=[]
            for k in range(len(gd_lon)):
                if sta_nas[k] not in rm_inx:
                # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        if self.tri:
                            if sta_tp[k]<9000 and sta_tp[k]>6000:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)

            label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
            ii+=1
            np.random.seed(ii)
            #=========================#
            if self.shift: 
                
                tmp=[]
                drop=False
                cn=len(gd_lon)
                rg=np.arange(cn)
                if cn>10:
                    drop=True  
                for k in rg[:cn]:
                    if sta_nas[k] not in rm_inx:
                        if drop:
                            if np.random.rand()<self.dp:
                                continue
                        # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]))
                        if self.tt==0:
                            sft=0  
                        else:
                            sft=int(np.random.uniform(-self.tt,self.tt,1)*100)
                        
                        sst=6000+sft
                        sed=sst+3000
                        if self.ave_single:
                            try:
                                mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                            except:
                                print(sta_tp[k])
                                mm=0
                        else:
                            mm=0
                        if mm!=0:
                            tmp=data[k,:,sst:sed:2]/mm
                        else:
                            tmp=data[k,:,sst:sed:2]
                        if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                            if self.tri:
                                if sta_tp[k]<9000+sft and sta_tp[k]>6000+sft:
                                    wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                            else:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0) 
                                
                            
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1              

            #=========================#
            if self.rot90:
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-1,::-1,:,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-1,::-1,:,:,0],axes=(1,0,2) )  
                ii+=1  
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-2,:,::-1,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-2,:,::-1,:,0],axes=(1,0,2) )  
                ii+=1                
                wave[ii,:,:,:,:] = np.transpose(wave[ii-3,::-1,::-1,:,:],axes=(0,1,2,3) )             
                label[ii,:,:,:,0] =  np.transpose(label[ii-3,::-1,::-1,:,0],axes=(0,1,2) ) 
                ii+=1                  

            #=========================#
            if self.rot:            

                an=np.pi/np.random.randint(4,10)*np.random.choice([-1,1])              
                gd_lon,gd_lat,gsi_lon,gsi_lat=self._rotate2(gd_lon,gd_lat,an,gsi_lon,gsi_lat)
                    
                tmp=[]
                for k in range(len(gd_lon)):
                    if sta_nas[k] not in rm_inx:
                    
                        # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        if self.ave_single:
                            try:
                                mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                            except:
                                print(sta_tp[k])
                                mm=0
                        else:
                            mm=0
                        if mm!=0:
                            tmp=data[k,:,6000:9000:2]/mm
                        else:
                            tmp=data[k,:,6000:9000:2]
                        if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                            # wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                            if self.tri:
                                if sta_tp[k]<9000 and sta_tp[k]>6000:
                                    wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                            else:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)                        
                        
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1            

        if self.mul:
            
            for i in range(len(batch_inds)):
                n1=np.random.randint(2,4)
                ex=[]
                for i1 in range(n1):
                    i2=np.random.randint(0,len(batch_inds))
                    
                    if i2 in ex:
                        if nn>4:
                            wave[ii,:,:,:,:]+=wave[i2*nn+4,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn+4,:,:,:,0]   
                        else:
                            i2=i2-1
                            ex.append(i2)
                            wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]                            
                    else:
                        ex.append(i2)
                        wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                        label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]
                        
                ii+=1

        # return wave,label    
        if self.ave_single:
            return wave,label  
        else:
            return self._normal3(wave),label          
        
        

        
# In[]
class DataGenerator_CULOC_test(Sequence):

    def __init__(self,df,fsc,si_id,si_unique,batch_size=128,w1=50,w2=50,x0=-99,y0=35,dx=20,
                 ave_single=False,shift=False,dp=0.1,tt=2,mul=False,tri=True,dr=4,
                 shuffle=True,rot=False,rot90=False,org=True):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
            
        self.batch_size = batch_size
        self.df=df
        self.fsc=fsc
        self.si_id=si_id
        self.si_unique=si_unique
        self.indexes=np.arange(len(self.si_id))
        self.shuffle = shuffle
        self.w1=w1
        self.w2=w2
        self.x0=x0
        self.y0=y0
        self.dx=dx
        self.rot=rot
        self.rot90=rot90
        self.shift=shift
        self.ave_single=ave_single
        self.dp=dp
        self.tt=tt
        self.mul=mul
        self.tri=tri
        self.dr=dr
        self.org=org

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.si_id)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        # read batch data
        X, Y= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
        return ({'input': X[:,:self.w1,:self.w2,:3000,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def _mk_lab(self,x,y,z,d=10):
        v=np.zeros((self.w1,self.w2,100))
        for i in range(self.w1):
            for j in range(self.w2):
                for k in range(100):
                    v[i,j,k]=np.exp(-((x-i*1.0)**2+(y-j*1.0)**2+(z-k*1.0)**2)/2/d**2)
                    #/(2*math.pi)**(3/2)/d**3
        return v
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1 
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
        
    def _rotate2(self,x,y,an,x1,y1):
        
        halfx = int( np.floor(self.w1 / 2) )
        halfy = int( np.floor(self.w2 / 2) )  
        x=[i-halfx for i in x]
        y=[i-halfy for i in y]
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y+halfx
        new_y=np.cos(an)*y+np.sin(an)*x+halfy
        # b_x1,b_x2 = np.min(new_x),np.max(new_x)
        # b_y1,b_y2 = np.min(new_y),np.max(new_y)
        x1=x1-halfx
        y1=y1-halfy
        s_x=np.cos(an)*x1-np.sin(an)*y1+halfx
        s_y=np.cos(an)*y1+np.sin(an)*x1 +halfy 
        
        new_x=[int(i) for i in new_x]
        new_y=[int(i) for i in new_y]
        # return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1    
        return new_x,new_y,int(s_x),int(s_y)   
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        for i in range(data.shape[0]):
            data1=data[i,:,:,:,:]
            x_max=np.max(np.max(abs(data1)))
            if x_max!=0.0:
                data2[i,:,:,:,:]=data1/x_max 
        return data2   
    
    def _read_data(self, batch_inds,si_id,si_unique,df,fsc):
        ll=0
        nn=0
        if self.org:
            ll+=len(batch_inds)
            nn+=1
        if self.rot:
            ll+=len(batch_inds)
            nn+=1
        if self.rot90:
            ll+=len(batch_inds)*3
            nn+=1
        if self.shift:
            ll+=len(batch_inds) 
            nn+=1
        if self.mul:
            ll+=len(batch_inds) 
            

        wave=np.zeros((ll,self.w1,self.w2,1500,3))
        label=np.zeros((ll,self.w1,self.w2,100,1))            
     
        ii=0     
        for i in batch_inds: 
            si_trace=df[(df.source_id==si_unique[si_id[i]])]['trace_name'].tolist()  
            data=[]
            for name in si_trace:
                temp=np.array(fsc.get('data/'+name))
                for iii in range(3):
                    temp[iii,:]=self._taper(temp[iii,:],1,100) 
                temp = self._bp_filter(temp,2,1,45,0.01)
                if len(temp[:,:36000][1])!=36000:
                    data.append(np.zeros((3,36000)))
                else:
                    data.append(temp[:,:36000]) 
            try:
                data=np.array(data) 
            except:
                print(si_unique[si_id[i]])
            
            sta_lon=df[(df.source_id==si_unique[si_id[i]])]['receiver_longitude'].tolist() 
            sta_lat=df[(df.source_id==si_unique[si_id[i]])]['receiver_latitude'].tolist() 
            sta_tp=df[(df.source_id==si_unique[si_id[i]])]['p_arrival_sample'].tolist()
            
            si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0] 
            si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]         
            si_dep=df[(df.source_id==si_unique[si_id[i]])]['source_depth_km'].tolist()[0] 
            gsi_dep=int(si_dep/1000)

            x0= self.x0
            y0= self.y0
            dx= self.dx
            gd_lon=[int((i1-x0)*dx) for i1 in sta_lon ]
            gd_lat=[int((i1-y0)*dx) for i1 in sta_lat ]
            
            gsi_lon=int((si_lon-x0)*dx)
            gsi_lat=int((si_lat-y0)*dx)  
            if self.org:
                tmp=[]
                for k in range(len(gd_lon)):
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        if self.tri:
                            if sta_tp[k]<9000:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
    
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1

            np.random.seed(ii)
            #=========================#
            if self.shift: 
                
                tmp=[]
                drop=False
                cn=len(gd_lon)
                rg=np.arange(cn)
                if cn>10:
                    drop=True
                    cn=int(cn*(1-self.dp))
                    np.random.shuffle(rg)   
                for k in rg[:cn]:
                    # if drop:
                    #     if np.random.rand()<self.dp:
                    #         continue
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]))
                    if self.tt==0:
                        sft=0  
                    else:
                        sft=int(np.random.uniform(-self.tt,self.tt,1)*100)
                    
                    sst=6000+sft
                    sed=sst+3000
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,sst:sed:2]/mm
                    else:
                        tmp=data[k,:,sst:sed:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        if self.tri:
                            if sta_tp[k]<9000+sft:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0) 
                            
                            
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1              

            #=========================#
            if self.rot90:
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-1,::-1,:,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-1,::-1,:,:,0],axes=(1,0,2) )  
                ii+=1  
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-2,:,::-1,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-2,:,::-1,:,0],axes=(1,0,2) )  
                ii+=1                
                wave[ii,:,:,:,:] = np.transpose(wave[ii-3,::-1,::-1,:,:],axes=(0,1,2,3) )             
                label[ii,:,:,:,0] =  np.transpose(label[ii-3,::-1,::-1,:,0],axes=(0,1,2) ) 
                ii+=1                  

            #=========================#
            if self.rot:            

                an=np.pi/np.random.randint(4,10)*np.random.choice([-1,1])              
                gd_lon,gd_lat,gsi_lon,gsi_lat=self._rotate2(gd_lon,gd_lat,an,gsi_lon,gsi_lat)
                    
                tmp=[]
                for k in range(len(gd_lon)):
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        # wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        if self.tri:
                            if sta_tp[k]<9000:
                                wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        else:
                            wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)                        
                        
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=self.dr)
                ii+=1            

        if self.mul:
            for i in range(len(batch_inds)):
                n1=np.random.randint(2,4)
                ex=[]
                for i1 in range(n1):
                    i2=np.random.randint(0,len(batch_inds))
                    
                    if i2 in ex:
                        if nn>4:
                            wave[ii,:,:,:,:]+=wave[i2*nn+4,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn+4,:,:,:,0]   
                        else:
                            i2=i2-1
                            ex.append(i2)
                            wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]                            
                    else:
                        ex.append(i2)
                        wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                        label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]
                        
                ii+=1

        # return wave,label    
        if self.ave_single:
            return wave,label  
        else:
            return self._normal3(wave),label    




# In[]
class DataGenerator_CULOC_Mul(Sequence):

    def __init__(self,df,fsc,si_id,si_unique,tt=5,dp=0.1,tri=True,
                 batch_size=128,w1=50,w2=50,x0=-99,y0=35,dx=20,shuffle=True,
                 rot=False,rot90=False,mul=False,shift=False,ave_single=False):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
            
        self.batch_size = batch_size
        self.df=df
        self.fsc=fsc
        self.si_id=si_id
        self.si_unique=si_unique
        self.indexes=np.arange(len(self.si_id))
        self.shuffle = shuffle
        self.w1=w1
        self.w2=w2
        self.x0=x0
        self.y0=y0
        self.dx=dx
        self.rot=rot
        self.rot90=rot90
        self.mul=mul
        self.shift=shift
        self.ave_single=ave_single
        self.dp=dp
        self.tt=tt
        self.tri=tri

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.si_id)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        # read batch data
        X, Y= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
        return ({'input': X[:,:self.w1,:self.w2,:,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def _mk_lab(self,x,y,z,d=10):
        v=np.zeros((self.w1,self.w2,100))
        for i in range(self.w1):
            for j in range(self.w2):
                for k in range(100):
                    v[i,j,k]=np.exp(-((x-i*1.0)**2+(y-j*1.0)**2+(z-k*1.0)**2)/2/d**2)
                    #/(2*math.pi)**(3/2)/d**3
        
        return v
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1 
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _rotate(self,x,y,an,x1,y1):
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y
        new_y=np.cos(an)*y+np.sin(an)*x
        b_x1,b_x2 = np.min(new_x),np.max(new_x)
        b_y1,b_y2 = np.min(new_y),np.max(new_y)
        
        s_x=np.cos(an)*x1-np.sin(an)*y1
        s_y=np.cos(an)*y1+np.sin(an)*x1  
        
        new_x=[int(i-b_x1+1) for i in new_x]
        new_y=[int(i-b_y1+1) for i in new_y]
        
        return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1
    
    def _rotate2(self,x,y,an,x1,y1):
        
        halfx = int( np.floor(self.w1 / 2) )
        halfy = int( np.floor(self.w2 / 2) )  
        x=[i-halfx for i in x]
        y=[i-halfy for i in y]
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y+halfx
        new_y=np.cos(an)*y+np.sin(an)*x+halfy
        # b_x1,b_x2 = np.min(new_x),np.max(new_x)
        # b_y1,b_y2 = np.min(new_y),np.max(new_y)
        x1=x1-halfx
        y1=y1-halfy
        s_x=np.cos(an)*x1-np.sin(an)*y1+halfx
        s_y=np.cos(an)*y1+np.sin(an)*x1 +halfy 
        
        new_x=[int(i) for i in new_x]
        new_y=[int(i) for i in new_y]
        # return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1    
        return new_x,new_y,int(s_x),int(s_y)
    
    
    
    def _normal33(self,data):
        return data/np.max(np.max(data))
        
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        for i in range(data.shape[0]):
            data1=data[i,:,:,:,:]
            x_max=np.max(np.max(abs(data1)))
            if x_max!=0.0:
                data2[i,:,:,:,:]=data1/x_max 
        return data2    

    def _read_data(self, batch_inds,si_id,si_unique,df,fsc):
        ll=len(batch_inds)
        nn=1
        if self.rot:
            ll+=len(batch_inds)
            nn+=1
        if self.shift:
            ll+=len(batch_inds) 
            nn+=1
        if self.mul:
            ll+=len(batch_inds)
        if self.rot90:
            ll+=len(batch_inds)*3
            nn+=3
            

        wave=np.zeros((ll,self.w1,self.w2,1500,3))
        label=np.zeros((ll,self.w1,self.w2,100,1))            
     
        ii=0
        for i in batch_inds:
        
            si_trace=df[(df.source_id==si_unique[si_id[i]])]['trace_name'].tolist()  
            data=[]
            for name in si_trace:
                temp=np.array(fsc.get('data/'+name))
                for iii in range(3):
                    temp[iii,:]=self._taper(temp[iii,:],1,100) 
                temp = self._bp_filter(temp,2,1,45,0.01)
                if len(temp[:,:36000][1])!=36000:
                    data.append(np.zeros((3,36000)))
                else:
                    data.append(temp[:,:36000]) 
            try:
                data=np.array(data) 
            except:
                print(si_unique[si_id[i]])
            
            sta_lon=df[(df.source_id==si_unique[si_id[i]])]['receiver_longitude'].tolist() 
            sta_lat=df[(df.source_id==si_unique[si_id[i]])]['receiver_latitude'].tolist() 
            sta_tp=df[(df.source_id==si_unique[si_id[i]])]['p_arrival_sample'].tolist()
            
            si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0] 
            si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]         
            si_dep=df[(df.source_id==si_unique[si_id[i]])]['source_depth_km'].tolist()[0] 
            gsi_dep=int(si_dep/1000)

            x0= self.x0
            y0= self.y0
            dx= self.dx
            gd_lon=[int((i1-x0)*dx) for i1 in sta_lon ]
            gd_lat=[int((i1-y0)*dx) for i1 in sta_lat ]
            
            gsi_lon=int((si_lon-x0)*dx)
            gsi_lat=int((si_lat-y0)*dx)  
            
            tmp=[]
            for k in range(len(gd_lon)):
                # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                if self.ave_single:
                    try:
                        mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    except:
                        print(sta_tp[k])
                        mm=0
                else:
                    mm=0
                if mm!=0:
                    tmp=data[k,:,6000:9000:2]/mm
                else:
                    tmp=data[k,:,6000:9000:2]
                if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                    wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                  
            label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
            ii+=1
            
            #=========================#
            if self.shift:  
                tmp=[]
                drop=False
                if len(gd_lon)>10:
                    drop=True
                for k in range(len(gd_lon)):
                    if drop:
                        if np.random.rand()<self.dp:
                            continue
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]))
                    if self.tt==0:
                        sft=0  
                    else:
                        sft=int(np.random.uniform(-self.tt,self.tt,1)*100)
                    
                    sst=6000+sft
                    sed=sst+3000
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,sst:sed:2]/mm
                    else:
                        tmp=data[k,:,sst:sed:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                      
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
                ii+=1            

            #=========================#
            if self.rot90:
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-1,::-1,:,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-1,::-1,:,:,0],axes=(1,0,2) )  
                ii+=1  
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-2,:,::-1,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-2,:,::-1,:,0],axes=(1,0,2) )  
                ii+=1                
                wave[ii,:,:,:,:] = np.transpose(wave[ii-3,::-1,::-1,:,:],axes=(0,1,2,3) )             
                label[ii,:,:,:,0] =  np.transpose(label[ii-3,::-1,::-1,:,0],axes=(0,1,2) ) 
                ii+=1                  

            #=========================#
            if self.rot:
                                
                np.random.seed(ii)
                an=np.pi/np.random.randint(4,10)*np.random.choice([-1,1])              
                gd_lon,gd_lat,gsi_lon,gsi_lat=self._rotate2(gd_lon,gd_lat,an,gsi_lon,gsi_lat)
                    
                tmp=[]
                for k in range(len(gd_lon)):
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
                ii+=1       
        if self.mul:
            for i in range(len(batch_inds)):
                # i1=np.random.randint(0,len(batch_inds))
                # i2=np.random.randint(0,len(batch_inds))
                # if i1==i2:
                #     i2=np.random.randint(0,len(batch_inds))
                # wave[ii,:,:,:,:]=wave[i1,:,:,:,:]+wave[i2,:,:,:,:]
                # label[ii,:,:,:,0]=label[i1,:,:,:,0]+label[i2,:,:,:,0]
                #================================================#
                n1=np.random.randint(2,4)
                ex=[]
                for i1 in range(n1):
                    i2=np.random.randint(0,len(batch_inds))
                    
                    if i2 in ex:
                        if nn>4:
                            wave[ii,:,:,:,:]+=wave[i2*nn+4,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn+4,:,:,:,0]   
                        else:
                            i2=i2-1
                            ex.append(i2)
                            wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]                            
                    else:
                        ex.append(i2)
                        wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                        label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]
                        
                ii+=1
        if self.ave_single:
            return wave,label
        else:
            return self._normal3(wave),self._normal3(label)
# In[]
class DataGenerator_CULOC_SCSN(Sequence):

    def __init__(self,df,fsc,si_id,si_unique,tt=5,dp=0.1,tri=True,
                 batch_size=128,w1=50,w2=50,x0=-99,y0=35,dx=20,shuffle=True,
                 rot=False,rot90=False,mul=False,shift=False,ave_single=False):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
            
        self.batch_size = batch_size
        self.df=df
        self.fsc=fsc
        self.si_id=si_id
        self.si_unique=si_unique
        self.indexes=np.arange(len(self.si_id))
        self.shuffle = shuffle
        self.w1=w1
        self.w2=w2
        self.x0=x0
        self.y0=y0
        self.dx=dx
        self.rot=rot
        self.rot90=rot90
        self.mul=mul
        self.shift=shift
        self.ave_single=ave_single
        self.dp=dp
        self.tt=tt
        self.tri=tri

    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.si_id)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        # read batch data
        X, Y= self._read_data(batch_inds,self.si_id,self.si_unique,self.df,self.fsc)
        return ({'input': X[:,:self.w1,:self.w2,:,:]}, {'output':Y[:,:self.w1,:self.w2,:,0] }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def _mk_lab(self,x,y,z,d=10):
        v=np.zeros((self.w1,self.w2,100))
        for i in range(self.w1):
            for j in range(self.w2):
                for k in range(100):
                    v[i,j,k]=np.exp(-((x-i*1.0)**2+(y-j*1.0)**2+(z-k*1.0)**2)/2/d**2)
                    #/(2*math.pi)**(3/2)/d**3
        
        return v
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1 
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _rotate(self,x,y,an,x1,y1):
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y
        new_y=np.cos(an)*y+np.sin(an)*x
        b_x1,b_x2 = np.min(new_x),np.max(new_x)
        b_y1,b_y2 = np.min(new_y),np.max(new_y)
        
        s_x=np.cos(an)*x1-np.sin(an)*y1
        s_y=np.cos(an)*y1+np.sin(an)*x1  
        
        new_x=[int(i-b_x1+1) for i in new_x]
        new_y=[int(i-b_y1+1) for i in new_y]
        
        return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1
    
    def _rotate2(self,x,y,an,x1,y1):
        
        halfx = int( np.floor(self.w1 / 2) )
        halfy = int( np.floor(self.w2 / 2) )  
        x=[i-halfx for i in x]
        y=[i-halfy for i in y]
        x=np.array(x)
        y=np.array(y)
        new_x=np.cos(an)*x-np.sin(an)*y+halfx
        new_y=np.cos(an)*y+np.sin(an)*x+halfy
        # b_x1,b_x2 = np.min(new_x),np.max(new_x)
        # b_y1,b_y2 = np.min(new_y),np.max(new_y)
        x1=x1-halfx
        y1=y1-halfy
        s_x=np.cos(an)*x1-np.sin(an)*y1+halfx
        s_y=np.cos(an)*y1+np.sin(an)*x1 +halfy 
        
        new_x=[int(i) for i in new_x]
        new_y=[int(i) for i in new_y]
        # return new_x,new_y,int(s_x-b_x1)+1,int(s_y-b_y1)+1    
        return new_x,new_y,int(s_x),int(s_y)
    
    
    
    def _normal33(self,data):
        return data/np.max(np.max(data))
        
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        for i in range(data.shape[0]):
            data1=data[i,:,:,:,:]
            x_max=np.max(np.max(abs(data1)))
            if x_max!=0.0:
                data2[i,:,:,:,:]=data1/x_max 
        return data2    

    def _read_data(self, batch_inds,si_id,si_unique,df,fsc):
        ll=len(batch_inds)
        nn=1
        if self.rot:
            ll+=len(batch_inds)
            nn+=1
        if self.shift:
            ll+=len(batch_inds) 
            nn+=1
        if self.mul:
            ll+=len(batch_inds)
        if self.rot90:
            ll+=len(batch_inds)*3
            nn+=3
            

        wave=np.zeros((ll,self.w1,self.w2,1500,3))
        label=np.zeros((ll,self.w1,self.w2,100,1))            
     
        ii=0
        for i in batch_inds:
        
            si_trace=df[(df.source_id==si_unique[si_id[i]])]['trace_name'].tolist()  
            data=[]
            for name in si_trace:
                temp=np.array(fsc.get('data/'+name))
                for iii in range(3):
                    temp[iii,:]=self._taper(temp[iii,:],1,100) 
                temp = self._bp_filter(temp,2,1,45,0.01)
                if len(temp[:,:36000][1])!=36000:
                    data.append(np.zeros((3,36000)))
                else:
                    data.append(temp[:,:36000]) 
            try:
                data=np.array(data) 
            except:
                print(si_unique[si_id[i]])
            
            sta_lon=df[(df.source_id==si_unique[si_id[i]])]['receiver_longitude'].tolist() 
            sta_lat=df[(df.source_id==si_unique[si_id[i]])]['receiver_latitude'].tolist() 
            sta_tp=df[(df.source_id==si_unique[si_id[i]])]['p_arrival_sample'].tolist()
            
            si_lon=df[(df.source_id==si_unique[si_id[i]])]['source_longitude'].tolist()[0] 
            si_lat=df[(df.source_id==si_unique[si_id[i]])]['source_latitude'].tolist()[0]         
            si_dep=df[(df.source_id==si_unique[si_id[i]])]['source_depth_km'].tolist()[0] 
            gsi_dep=int(si_dep/1000)

            x0= self.x0
            y0= self.y0
            dx= self.dx
            gd_lon=[int((i1-x0)*dx) for i1 in sta_lon ]
            gd_lat=[int((i1-y0)*dx) for i1 in sta_lat ]
            
            gsi_lon=int((si_lon-x0)*dx)
            gsi_lat=int((si_lat-y0)*dx)  
            
            tmp=[]
            for k in range(len(gd_lon)):
                # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                if self.ave_single:
                    try:
                        mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    except:
                        print(sta_tp[k])
                        mm=0
                else:
                    mm=0
                if mm!=0:
                    tmp=data[k,:,6000:9000:2]/mm
                else:
                    tmp=data[k,:,6000:9000:2]
                if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                    wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                  
            label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
            ii+=1
            
            #=========================#
            if self.shift:  
                tmp=[]
                drop=False
                if len(gd_lon)>10:
                    drop=True
                for k in range(len(gd_lon)):
                    if drop:
                        if np.random.rand()<self.dp:
                            continue
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]))
                    if self.tt==0:
                        sft=0  
                    else:
                        sft=int(np.random.uniform(-self.tt,self.tt,1)*100)
                    
                    sst=6000+sft
                    sed=sst+3000
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,sst:sed:2]/mm
                    else:
                        tmp=data[k,:,sst:sed:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                      
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
                ii+=1            

            #=========================#
            if self.rot90:
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-1,::-1,:,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-1,::-1,:,:,0],axes=(1,0,2) )  
                ii+=1  
                wave[ii,:,:,:,:] =  np.transpose(  wave[ii-2,:,::-1,:,:],axes=(1,0,2,3) )              
                label[ii,:,:,:,0] =  np.transpose(  label[ii-2,:,::-1,:,0],axes=(1,0,2) )  
                ii+=1                
                wave[ii,:,:,:,:] = np.transpose(wave[ii-3,::-1,::-1,:,:],axes=(0,1,2,3) )             
                label[ii,:,:,:,0] =  np.transpose(label[ii-3,::-1,::-1,:,0],axes=(0,1,2) ) 
                ii+=1                  

            #=========================#
            if self.rot:
                                
                np.random.seed(ii)
                an=np.pi/np.random.randint(4,10)*np.random.choice([-1,1])              
                gd_lon,gd_lat,gsi_lon,gsi_lat=self._rotate2(gd_lon,gd_lat,an,gsi_lon,gsi_lat)
                    
                tmp=[]
                for k in range(len(gd_lon)):
                    # tmp=data[k,:,sta_tp[k]-1000:sta_tp[k]+5000]/np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                    if self.ave_single:
                        try:
                            mm=np.max(abs(data[k,:,sta_tp[k]-1000:sta_tp[k]+5000])) 
                        except:
                            print(sta_tp[k])
                            mm=0
                    else:
                        mm=0
                    if mm!=0:
                        tmp=data[k,:,6000:9000:2]/mm
                    else:
                        tmp=data[k,:,6000:9000:2]
                    if gd_lon[k]>0 and gd_lon[k]<self.w1 and gd_lat[k]>0and gd_lat[k]<self.w2:
                        wave[ii,gd_lon[k],gd_lat[k],:,:]=tmp.transpose(1,0)
                        
                label[ii,:,:,:,0]=self._mk_lab(gsi_lon,gsi_lat,gsi_dep,d=4)
                ii+=1       
        if self.mul:
            for i in range(len(batch_inds)):
                # i1=np.random.randint(0,len(batch_inds))
                # i2=np.random.randint(0,len(batch_inds))
                # if i1==i2:
                #     i2=np.random.randint(0,len(batch_inds))
                # wave[ii,:,:,:,:]=wave[i1,:,:,:,:]+wave[i2,:,:,:,:]
                # label[ii,:,:,:,0]=label[i1,:,:,:,0]+label[i2,:,:,:,0]
                #================================================#
                n1=np.random.randint(2,4)
                ex=[]
                for i1 in range(n1):
                    i2=np.random.randint(0,len(batch_inds))
                    
                    if i2 in ex:
                        if nn>4:
                            wave[ii,:,:,:,:]+=wave[i2*nn+4,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn+4,:,:,:,0]   
                        else:
                            i2=i2-1
                            ex.append(i2)
                            wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                            label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]                            
                    else:
                        ex.append(i2)
                        wave[ii,:,:,:,:]+=wave[i2*nn,:,:,:,:]
                        label[ii,:,:,:,0]+=label[i2*nn,:,:,:,0]
                        
                ii+=1
        if self.ave_single:
            return wave,label
        else:
            return self._normal3(wave),self._normal3(label)

# In[]  
def plot_3d_1(data,name='Data',fl=0,x0=-99, x1=-96.5, y0=35, y1=37.5):
    
    font2 = {'family' : 'Times New Roman',    'weight' : 'normal',    'size'   : 18,    }    
    
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[:-1, :-1])
    y_hist = fig.add_subplot(grid[:-1, -1], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, :-1], sharex=main_ax)
    #
    main_ax.imshow(np.max(data,axis=2))#,origin='lower',extent=(x0, x1, y0, y1))
    evt=np.where(data==np.max(data))
    if fl==1:
        main_ax.plot(evt[1][0],evt[0][0],'r',marker = '*',markersize=15)

    main_ax.set_title(name,font2)
    main_ax.tick_params(labelsize=0)
    x_hist.imshow(np.max(data,axis=0).T,aspect='auto')#,extent=(x0, x1, 0, 50))
    if fl==1:
        x_hist.plot(evt[1][0],evt[2][0],'r',marker = '*',markersize=15)    
    
    
    # x_hist.invert_yaxis()
    x_hist.set_xlabel('Longitude',font2)
    x_hist.yaxis.tick_right()
    x_hist.tick_params(labelsize=15)
    
    y_hist.imshow(np.max(data,axis=1),aspect='auto')#,extent=(0, 50, y0, y1))
    if fl==1:
        y_hist.plot(evt[2][0],evt[0][0],'r',marker = '*',markersize=15)        
    
    # y_hist.invert_xaxis()
    y_hist.set_xlabel('Depth',font2)
    y_hist.yaxis.tick_right()
    y_hist.set_ylabel('Latitude',font2)
    y_hist.yaxis.set_label_position("right")
    y_hist.tick_params(labelsize=15)
    
    plt.show()    
# In[]  
def eva_model(data,r=0.5,n=9):
    oup=np.max(data,axis=2) 
    bxy=[]
    # main_ax.plot(evt[1][0],evt[0][0],'r',marker = '*',markersize=15)
    contours = measure.find_contours(oup/np.max(oup), r,fully_connected='low')
    for n, contour in enumerate(contours):
        if (int(np.min(contour[:, 0]))-int(np.max(contour[:, 0])))*(int(np.min(contour[:, 1]))-int(np.max(contour[:, 1])))<n:
            continue
        bxy.append([int(np.min(contour[:, 0])),int(np.max(contour[:, 0])),int(np.min(contour[:, 1])),int(np.max(contour[:, 1]))])
    res=[]
    for i in range(len(bxy)):
        # print(bxy[i][0],bxy[i][1],bxy[i][2],bxy[i][3])
        try:
            evt1=np.where(data[bxy[i][0]:bxy[i][1],bxy[i][2]:bxy[i][3],:]==np.max(data[bxy[i][0]:bxy[i][1],bxy[i][2]:bxy[i][3],:]))    
            # print(bxy[i][2]+evt1[1][0],bxy[i][0]+evt1[0][0],evt1[2][0])
            res.append([bxy[i][2]+evt1[1][0],bxy[i][0]+evt1[0][0],evt1[2][0]])
        except:
            res.append([-1,-1,-1])
    return res
    
    

    
# In[]  
def plot_3d(data,name='Data',fl=0,r=0.5,path=None):
    
    font2 = {'family' : 'Times New Roman',    'weight' : 'normal',    'size'   : 18,    }   

    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[:-1, :-1])
    # y_hist = fig.add_subplot(grid[:-1, -1], xticklabels=[], sharey=main_ax)
    # x_hist = fig.add_subplot(grid[-1, :-1], yticklabels=[], sharex=main_ax)
    y_hist = fig.add_subplot(grid[:-1, -1], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, :-1], sharex=main_ax)
    oup=np.max(data,axis=2)    
    main_ax.imshow(oup,origin='lower')
    evt=np.where(data==np.max(data))
    
    if fl==1:
        bxy=[]
        # main_ax.plot(evt[1][0],evt[0][0],'r',marker = '*',markersize=15)
        contours = measure.find_contours(oup/np.max(oup), r,fully_connected='low')
        for n, contour in enumerate(contours):
            main_ax.plot(contour[:, 1], contour[:, 0], 'r',linewidth=2)
            if (int(np.min(contour[:, 0]))-int(np.max(contour[:, 0])))*(int(np.min(contour[:, 1]))-int(np.max(contour[:, 1])))<9:
                continue
            
            bxy.append([int(np.min(contour[:, 0])),int(np.max(contour[:, 0])),int(np.min(contour[:, 1])),int(np.max(contour[:, 1]))])
        
        d0=np.max(data,axis=2)
        for i in range(len(bxy)):
            print(bxy[i][0],bxy[i][1],bxy[i][2],bxy[i][3])
            evt1=np.where(d0[bxy[i][0]:bxy[i][1],bxy[i][2]:bxy[i][3]]==np.max(d0[bxy[i][0]:bxy[i][1],bxy[i][2]:bxy[i][3]]))
            main_ax.plot(bxy[i][2]+evt1[1][0],bxy[i][0]+evt1[0][0],'r',marker = '*',markersize=15)  
        
    main_ax.set_title(name,font2)
    main_ax.tick_params(labelsize=0)
    # main_ax.set_xticks([])
    # main_ax.set_yticks([])
    x_hist.imshow(np.max(data,axis=0).T,aspect='auto')
    if fl==1:
        # x_hist.plot(evt[1][0],evt[2][0],'r',marker = '*',markersize=15)  
        d1=np.max(data,axis=0)
        for i in range(len(bxy)):
            print(bxy[i][2],bxy[i][3])
            evt1=np.where(d1[bxy[i][2]:bxy[i][3],:]==np.max(d1[bxy[i][2]:bxy[i][3],:]))
            x_hist.plot(bxy[i][2]+evt1[0][0],evt1[1][0],'r',marker = '*',markersize=15)  

    # x_hist.invert_yaxis()
    x_hist.set_xlabel('X',font2)
    x_hist.yaxis.tick_right()
    x_hist.tick_params(labelsize=15)
    
    y_hist.imshow(np.max(data,axis=1),aspect='auto')
    if fl==1:
        # y_hist.plot(evt[2][0],evt[0][0],'r',marker = '*',markersize=15)  
        d2=np.max(data,axis=1)
        for i in range(len(bxy)):
            print(bxy[i][0],bxy[i][1])
            evt2=np.where(d2[bxy[i][0]:bxy[i][1],:]==np.max(d2[bxy[i][0]:bxy[i][1],:]))
            print(evt2)
            y_hist.plot(evt2[1][0],bxy[i][0]+evt2[0][0],'r',marker = '*',markersize=15)          

    # y_hist.invert_xaxis()
    y_hist.set_xlabel('Z',font2)
    y_hist.yaxis.tick_right()
    y_hist.set_ylabel('Y',font2)
    y_hist.yaxis.set_label_position("right")
    y_hist.tick_params(labelsize=15)
    if path is not None:
        plt.savefig(path,dpi=600)     
    plt.show()    
    
def plt_3d_wave(wave_inp,evt,i,path=None):
    font2={'family':'Times New Roman','weight':'bold','size':15}    
    inp=np.sum(abs(wave_inp[i,:,:,:,0]),axis=2)
    gd_lon,gd_lat=np.nonzero(inp)
    figure  = plt.figure(figsize=(12, 7))
    ax1 = plt.axes(projection='3d')
    z=np.arange(1500,)
    for j in range(len(gd_lon)):
        x=gd_lon[j]+wave_inp[i,gd_lon[j],gd_lat[j],:,0]/np.max(abs(wave_inp[i,gd_lon[j],gd_lat[j],:,0]))
        y=gd_lat[j]+np.arange(1500,)*0
        ax1.plot(x, y, z,color='k')
    try:     
        ax1.scatter(evt[1],evt[0],evt[2],c='r',s=200,marker='*')  
    except:
        evt=np.array(evt)
        ax1.scatter(evt[:,1],evt[:,0],evt[:,2],c='r',s=200,marker='*') 
   
    ax1.set_xlim(0,50)
    ax1.set_ylim(0,50)
    ax1.set_zlim(0,1500)
    ax1.set_xlabel('X',font2)
    ax1.set_ylabel('Y',font2)
    ax1.set_zlabel('Samples',font2)
    ax1.invert_zaxis()
    ax1.grid(ls='-')
    ax1.view_init(20,60)
    plt.tick_params(labelsize=15)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]    

    if path is not None:
        plt.savefig(path,dpi=600) 
    plt.show()
    
def plt_3d_loc(wave_oup,evt,i,path=None):
    font2={'family':'Times New Roman','weight':'bold','size':15}    

    figure  = plt.figure(figsize=(12, 7))
    ax1 = plt.axes(projection='3d') 
    
    X, Y = np.meshgrid(np.linspace(0,1,21), np.linspace(0,1,21))
    
    ax1.contourf(X, Y, wave_oup[i,:,:,:,0], 100, zdir='z', offset=0.5)
    # ax1.scatter(evt[0],evt[1],evt[2],c='r',s=200,marker='*')  
   
    ax1.set_xlim(0,50)
    ax1.set_ylim(0,50)
    ax1.set_zlim(0,50)
    ax1.set_xlabel('X',font2)
    ax1.set_ylabel('Y',font2)
    ax1.set_zlabel('Z',font2)
    ax1.invert_zaxis()
    ax1.grid(ls='-')
    ax1.view_init(20,60)
    plt.tick_params(labelsize=15)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]    

    if path is not None:
        plt.savefig(path,dpi=600) 
    plt.show()    
    
# In[]
def plot_loss(history_callback,save_path=None,model='model',c='c'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

    history_dict=history_callback.history

    loss_value=history_dict['loss']
    val_loss_value=history_dict['val_loss']

    try:
        acc_value=history_dict['binary_accuracy']
        val_acc_value=history_dict['val_binary_accuracy']
    except:
        acc_value=history_dict['accuracy']
        val_acc_value=history_dict['val_accuracy']
        
        

    epochs=range(1,len(acc_value)+1)
    if not save_path is None:
        np.savez(save_path+'acc_loss_%s'%model,
                 loss=loss_value,val_loss=val_loss_value,
                 acc=acc_value,val_acc=val_acc_value)


    figure, ax = plt.subplots(figsize=(8,6))

    if c=='k':
        plt.plot(epochs,acc_value,'k',label='Training acc')
        plt.plot(epochs,val_acc_value,'k-.',label='Validation acc')        
    else:
        plt.plot(epochs,acc_value,'r',label='Training acc')
        plt.plot(epochs,val_acc_value,'b-.',label='Validation acc')    
    
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_%s.png'%model,dpi=600)
    plt.show()

    figure, ax = plt.subplots(figsize=(8,6))
    if c=='k':
        plt.plot(epochs,loss_value,'k',label='Training loss')
        plt.plot(epochs,val_loss_value,'k-.',label='Validation loss')        
    else: 
        plt.plot(epochs,loss_value,'r',label='Training loss')
        plt.plot(epochs,val_loss_value,'b-.',label='Validation loss')    
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)    
    plt.show()    
    

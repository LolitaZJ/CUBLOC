# CUBLOC
A 3D U-Net Network to Achieve Earthquake Location       
Author: Ji Zhang  
Date: 2024.6.6  
Version 1.0.0  

# CUBLOC   
## Multi-station seismic location based on 3D U-Net

### This repository contains the codes to train and test the network proposed in:             

`Zhang J et al., Multi-station seismic location based on 3D U-Net, 2023.`
      
------------------------------------------- 
### Installation:

`pip install -r requirements.txt`
   
------------------------------------------- 
### Short Description:

Locating earthquakes plays an important role in the study of seismic activity and geological structures. Machine learning methods have been often applied to locate earthquakes. However, current machine learning approaches may face challenges in physical constraints and interpretability. Here, we build a 3D U-Net network with station distribution constraints to locate the earthquakes. The testing results show that this new method is stable and generalized, which should be applicable to earthquake location associated with arbitrary station networks.

------------------------------------------- 
### Dataset:

When you use this method to run your data, the format of the data is as consistent as possible with [STEAD](https://github.com/smousavi05/STEAD) (https://github.com/smousavi05/STEAD) or rewrite the generator for the dataset.

You can use [QuakeLabeler](https://maihao14.github.io/QuakeLabeler/) (https://maihao14.github.io/QuakeLabeler/) or [SeisBench](https://github.com/seisbench/seisbench) (https://github.com/seisbench/seisbench) to labele and convert your data into STEAD format. 

1. csv file: contains Event information,Station information,Trace information and so on; 
2. hdf5 file: contains Waveform dataset,Trace information;

------------------------------------------- 
### Parameters
Before you run this method, you need determine the information of your study area.

`--s_range [-99.0,-96.5,35.0,37.5]` 
The area in [min_lon,max_lon,min_lat,max_lat];      

`--grid [50,50]`
The number of the grid [lo,la] at longitude and latitude;
Recommend that they be equal in size.

`--dx 20` 
The size of the grid is 111/dx (km);

`--dr 2`
The size of Gaussain Radius is dr grids;

`--input_size (50,50,1500,3)`
The input size (lo,la,time series,ENZ),the 30s data (sampling 25Hz); 

`--data_path`
The path of hdf5 file

`--csv_path`
The path of csv file

`--save_path`
The path of save results

`--save_file`
The path of save figures

`--model_name`
The model name


------------------------------------------- 
### Run

` Train`
>     python CUBLOC_MASTER_MAIN.py --mode='train' --s_range [x1,x2,y1,y2] --data_path /your/hdf5/data/path/ --csv_path /your/csv/data/path/ --model_name CUBLOC_M01

`Test`
>     python CUBLOC_MASTER_MAIN.py --mode='test' --s_range [x1,x2,y1,y2] --data_path /your/hdf5/data/path/ --data_path /your/csv/data/path/ --model_name CUBLOC_M01

------------------------------------------- 
### Model
####Parameter of model####
input size (50,50,1500,3)   
dx=20  
dr=2   
grid=[50,50] 
####Trained model####
This model with no data augmentation   
`./CUBLOC_MASTER/GIT_LOC/CUBLOC_M01_r2.h5`

------------------------------------------- 

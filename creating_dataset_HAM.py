# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:41:37 2019

@author: akhilsantha

Script to work on the raw HAM-10000 dataset.
This organizes images of different classes into seperate folders (7 classes)

"""
import numpy as np
import pandas as pd
import os
import shutil

colnames = []

data = pd.read_csv('HAM10000_metadata.csv')

diseases = data.loc[:,'dx'] #loc of the column name
filenames = data.loc[:,'image_id']

classes = diseases.unique()

for disease in classes:
    #get all filenames for specific disease 
    files = filenames[diseases == disease]
    
    (num,) = files.shape 
    jpg = ['.jpg']
    lis = jpg*num
    p = files.str.cat(lis)
    
    #go through all the images
    for i in range(p.size):
        src_path = os.path.join('C:\Users\\akhilsantha\Desktop\HAM_dataset_work',p.iloc[i])
        des_path = os.path.join('C:\Users\\akhilsantha\\Desktop\HAM_dataset_work',disease)
        
        shutil.move(src_path,des_path)
    

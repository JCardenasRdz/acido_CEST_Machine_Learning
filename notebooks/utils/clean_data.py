#!/usr/bin/env python
# coding: utf-8
# Modules
import os
from   glob import glob
import numpy as np
import pandas as pd
import fastparquet # this is just to let pipreqs known we need this

# functions
def clean_my_file(file_name):
    """[summary]

    :param file_name: [description]
    :type file_name: [type]
    :return: [description]
    :rtype: [type]
    """
    
    # read csv
    exp = pd.read_csv(file_name)
    # list of freqs
    
    list_of_freqs = exp.columns[12::].to_list()
    
    # index of first real freq
    index = list_of_freqs.index('-12')
    
    # normalize
    norm_factor = exp[ list_of_freqs[0:index-1] ].iloc[:,-1]
    
    raw_data = exp[ list_of_freqs[index::] ]
    
    # preallocate with zeros
    Zspectra = np.zeros(raw_data.shape)
    
    
    for i in range(raw_data.shape[0]):
        Zspectra[i,:] = raw_data.iloc[i,:] / norm_factor[i]
        
    
    #out[exp.columns[0:11].to_list()] = exp.columns[0:11].copy()
    
    out = exp[ exp.columns[0:11] ].copy()
    
    out['FILE']  = file_name.split('/raw_data/')[1]
    
    Z = pd.DataFrame(Zspectra, columns= list_of_freqs[index::])
    
    out[Z.columns] = Z.copy()
    
    return out

# Load data
PATH = "../../raw_data"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

data = pd.DataFrame()

for file in all_csv_files:
    exp = clean_my_file(file)
    data = pd.concat( (data,  exp), sort=False )

print(f'shape of the original data: {data.shape}')

# round data
data['pH']     = data['pH'].apply(lambda x: np.round(x,1)) 
data['Conc(mM)']     = data['Conc(mM)'].apply(lambda x: np.round(x,1)) 
data['SatPower(uT)'] = data['SatPower(uT)'].apply(lambda x: np.round(x,1)) 
data['SatTime(ms)']  = data['SatTime(ms)'].apply(lambda x: np.round(x,1)) 


# keep data 
f1 = data['Conc(mM)']     > 5
f2 = data['SatPower(uT)'] > 0.5
f3 = data['SatTime(ms)']  > 500

# Save data
path = '../../clean_data/acido_CEST_MRI_Iopamidol.parquet.gzip'
data[f3&f2&f1].to_parquet(path, compression='gzip')
print(f'Done saving onto {path}')

### Code to Identify AlphaNumeric Features ###


import pandas as pd
import numpy as np
import os
import configparser
import re
import glob
from dateutil.parser import parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array

### Calling Configuration Files ###

config = configparser.ConfigParser()
config.read('regression_config.ini')


### Reading Config Data ###

data_path      =  config['DIRECTORY PATH']['data_path'] 
directory_path =  config['DIRECTORY PATH']['directory_path']
write_path     =  config['DIRECTORY PATH']['write_path_extractalpha']
machine_name   =  config['MACHINE NAME TYPE']['name']


os.chdir(directory_path)
if not os.path.exists(write_path):
    os.makedirs(write_path)

file_list = glob.glob(data_path+machine_name+"*.csv")

### Read First File ###

test_file = pd.read_csv(file_list[0])  
try:
    test_file.drop('Unnamed: 0', axis = 1,inplace = True)
except ValueError:
    pass

def hasNumbers(inputString):
    
    return any(char.isdigit() for char in inputString)
def remove_numbers(s):
    
    return ''.join([i for i in s if not i.isdigit()])


col_list = test_file.columns
alphanumeric_columns = [] 

for i in  range(0,len(col_list)):
    if hasNumbers(col_list[i]) == True:
        alphanumeric_columns.append(col_list[i])
 

no_nums = []

for i in alphanumeric_columns:

    no_nums.append(remove_numbers(i))

df_alphanumeric = pd.DataFrame({'var': alphanumeric_columns, 'non_num':no_nums})
df_alphanumeric.to_csv('AlphaNumericColumn.csv')

alphanum_maps =  list(df_alphanumeric.groupby(df_alphanumeric['non_num']))


nl = []
ol = []

for i in range(len(alphanum_maps)):
    nl.append(list(alphanum_maps[i][1]['var']))
    ol.append(alphanum_maps[i][0])

NumericColumnData = pd.DataFrame({'OriginalFeature':ol,'NumericFeature':nl})

NumericColumnData.to_csv(write_path+"AlphaNumericColumnData.csv")
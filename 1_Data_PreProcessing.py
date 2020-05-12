#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Team 18

"""

###### importing necessary packages #####
import pandas as pd
from scipy.io import arff ### since the file is in arff format #####

###### dataset preprocessing ##########
data = arff.loadarff('OBS-Network-DataSet_2_Aug27.arff')
df = pd.DataFrame(data[0])
df.shape
df.describe()
####### Since the column are not named properly  so column renaming is required #########
columns_new=['Node','Utilised_Bandwith_Rate','Packet_Drop_Rate','Full_Bandwidth',
         'Average_Delay_Time_Per_Sec','Percentage_Of_Lost_Packet_Rate','Percentage_Of_Lost_Byte_Rate','Packet_Received_Rate',
         'of_Used_Bandwidth','Lost_Bandwidth','Packet_Size_Byte','Packet_Transmitted',
         'Packet_Received','Packet_lost','Transmitted_Byte','Received_Byte',
         '10_Run_AVG_Drop_Rate','10_Run_AVG_Bandwith_Use','10_Run_Delay','Node_Status',
         'Flood_Status','Class']
len(columns_new)
df.columns=columns_new
df.columns
df.info
df.describe()
################################
###### checking for missing values #######
missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/1075)*100})
missing_data
######### removing rows having null entries ######
df = df[pd.notnull(df['Packet_lost'])]

######## cross checking after removing null values ######

missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/1075)*100})
missing_data
df.shape
df.columns
############ Plotting  HIstogram #################
hist_se=df.hist(bins=10, figsize=(20, 15),grid=False)
###### from histogram we can see that the feature 'Node' and packet size byte are of no use ###
######### droping column which are of no use #######
df1=df.drop(columns=['Packet_Size_Byte','Node'])
df1.shape

df1.head
######### interchanging feature 'Flood_Status' and 'Node_Status' #######
df=df1.reindex(['Utilised_Bandwith_Rate', 'Packet_Drop_Rate', 'Full_Bandwidth',
       'Average_Delay_Time_Per_Sec', 'Percentage_Of_Lost_Packet_Rate',
       'Percentage_Of_Lost_Byte_Rate', 'Packet_Received_Rate',
       'of_Used_Bandwidth', 'Lost_Bandwidth', 'Packet_Transmitted',
       'Packet_Received', 'Packet_lost', 'Transmitted_Byte', 'Received_Byte',
       '10_Run_AVG_Drop_Rate', '10_Run_AVG_Bandwith_Use', '10_Run_Delay',
       'Flood_Status', 'Node_Status', 'Class'],axis=1)

df.shape
df.head(5)
######## Converting column values of 'Node_status' and 'Class' in nummerical values ######
df['Node_Status'].unique()
df['Class'].unique()                       
df['Node_Status']=df['Node_Status'].map({"b'B'":0,'b"\'P NB\'"':1,"b'NB'":2})
df['Class']=df['Class'].map({'b"\'No Block\'"':0,'b"\'NB-No Block\'"':1,"b'NB-Wait'":2,"b'Block'":3})
df.head(5)

############ plotting to check for Outlier using box plot #######
df.to_csv('Dataset_Rearranged.csv', index=False)











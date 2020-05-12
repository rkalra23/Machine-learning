"""

@author: Tenm_18

"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

pd.options.display.float_format = "{:.6f}".format
df = pd.read_csv('Dataset_Rearranged.csv')
df.columns
df.head
df.describe()
######### checking for outlier using boxplot #######  
sns.boxplot(x=df['Utilised_Bandwith_Rate'])
sns.boxplot(x=df['Packet_Drop_Rate'])
sns.boxplot(x=df['Full_Bandwidth'])
sns.boxplot(x=df['Average_Delay_Time_Per_Sec'])
sns.boxplot(x=df['Percentage_Of_Lost_Packet_Rate'])
sns.boxplot(x=df['Percentage_Of_Lost_Byte_Rate'])
sns.boxplot(x=df['Packet_Received_Rate'])
sns.boxplot(x=df['of_Used_Bandwidth'])
sns.boxplot(x=df['Lost_Bandwidth'])
sns.boxplot(x=df['Packet_Transmitted'])
sns.boxplot(x=df['Packet_Received'])
sns.boxplot(x=df['Packet_lost'])
sns.boxplot(x=df['Transmitted_Byte'])
sns.boxplot(x=df['Received_Byte'])
sns.boxplot(x=df['10_Run_AVG_Drop_Rate'])
sns.boxplot(x=df['10_Run_AVG_Bandwith_Use'])
sns.boxplot(x=df['10_Run_Delay'])
sns.boxplot(x=df['Flood_Status'])
sns.boxplot(x=df['Node_Status'])

######### using z score, we will drop all the rows where the values are above ##
######### 3 Standard deviation from the mean                                  ##
z = np.abs(stats.zscore(df))
print(z)
threshold = 3
thresh_greater=np.where(z > 3)
len(thresh_greater)
print(np.where(z > 3))

df = df[(z < 3).all(axis=1)]
df
########### transporting dataframe to .csv and saving it ###########
df.to_csv('Dataset_Without_Outliers.csv', index=False)





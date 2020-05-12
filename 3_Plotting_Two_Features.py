"""

@author: Team_18

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

################################################
pd.options.display.float_format = "{:.6f}".format
df = pd.read_csv('Dataset_Without_Outliers.csv')
df.columns
df.head(5)

######## reading two features ######
X=df.iloc[:,1:3]
X.head(5)
####### transforming using standard scalar ##########
X = StandardScaler().fit_transform(X)

DF=pd.DataFrame(data = X,columns=['Packet_Drop_Rate','Full_Bandwidth'])

finalDf = pd.concat([DF, df[['Class']]], axis = 1)
finalDf.head

###### Plotting two features ##############
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Packet_Drop_Rate', fontsize = 15)
ax.set_ylabel('Full_Bandwidth', fontsize = 15)
ax.set_title('two features', fontsize = 20)
targets = [0,1,2,3]
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Packet_Drop_Rate']
               , finalDf.loc[indicesToKeep, 'Full_Bandwidth']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()




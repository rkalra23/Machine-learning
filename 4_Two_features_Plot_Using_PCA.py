"""

@author: Team_18

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA

################################################
pd.options.display.float_format = "{:.6f}".format
df = pd.read_csv('Dataset_Without_Outliers.csv')
df.columns
df.head
df[['Class']]
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Class']]], axis = 1)
finalDf.head
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('principal component 1', fontsize = 15)
ax.set_ylabel('principal component 2', fontsize = 15)
ax.set_title('two features', fontsize = 20)
targets = [0,1,2,3]
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()




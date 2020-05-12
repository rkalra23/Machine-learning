"""

@author: Team_18

"""

import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
import time
import os
import psutil
import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:.6f}".format
df = pd.read_csv('Dataset_Without_Outliers.csv')
df.shape
df.head(5)

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.3, random_state=50)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

########### using PCA for feature extraction ###########

pca = PCA(.95)
pca.fit(X_train) 

X_train = pca.transform(X_train)

X_test =pca.transform(X_test)

start = time.time()
svclassifier = SVC(kernel='rbf')  
accuracy = model_selection.cross_val_score(svclassifier, X_train, Y_train, scoring='accuracy', cv = 10)
print(accuracy)

print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)

svclassifier.fit(X_train,Y_train)
predictions=svclassifier.predict(X_test)
 
print(confusion_matrix(Y_test, predictions))  
print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

end = time.time()
print('Time complexity for SVN PCA base model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for SVN PCA base model is',process.memory_info().rss, 'bytes')


####### using grid search For hyperparameter tunning ######


 
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10,100,1000,10000]
    gammas = [0.001, 0.01, 0.1, 1,10,100,1000]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


bestparameterSVM=svc_param_selection(X_train,Y_train,5)

start = time.time()
svc=SVC(kernel='rbf',C=10000,gamma=0.1)

accuracy = model_selection.cross_val_score(svc, X_train, Y_train, scoring='accuracy', cv = 10)
print(accuracy)
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)


svc.fit(X_train,Y_train)
predictions=svc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score 
print(confusion_matrix(Y_test, predictions))  
print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

end = time.time()
print('Time complexity for SVN PCA optimized model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for SVN PCA optimized model is',process.memory_info().rss, 'bytes')




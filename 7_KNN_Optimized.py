"""

@author: Team_18

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.30, random_state=50)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_train.shape
X_test = scaler.transform(X_test)
X_test.shape

start = time.time()
Knnclassifier = KNeighborsClassifier() 
accuracy = model_selection.cross_val_score(Knnclassifier, X_train, Y_train, scoring='accuracy', cv = 10)
print(accuracy)
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100) 

Knnclassifier.fit(X_train, Y_train) 
y_pred = Knnclassifier.predict(X_test)  
print(confusion_matrix(Y_test, y_pred))  
print(classification_report(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))
end = time.time()
print('Time complexity for KNN base model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for KNN base model is',process.memory_info().rss, 'bytes')
   
########################## optimizing ###############################

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    
    # creating the KNN classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    # fitting the model
    clf.fit(X_train, Y_train)
    #recording the accuracy of the training set
    training_accuracy.append(clf.score(X_train, Y_train))
    #recording the accuracy of the test set
    test_accuracy.append(clf.score(X_test, Y_test))

plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the Training Set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the Test Set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()


'''
error = []

# Calculating error for K values between 1 and 40
for i in range(1,10):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')

'''

start = time.time()
Knnclassifier_optimized = KNeighborsClassifier(n_neighbors=4)

accuracy = model_selection.cross_val_score(Knnclassifier_optimized, X_train, Y_train, scoring='accuracy', cv = 10)
print(accuracy)
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)



Knnclassifier_optimized.fit(X_train,Y_train)
predictions=Knnclassifier_optimized.predict(X_test)

print(confusion_matrix(Y_test, predictions))  
print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

end = time.time()
print('Time complexity for KNN optimized model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for KNN optimized model is',process.memory_info().rss, 'bytes')
 








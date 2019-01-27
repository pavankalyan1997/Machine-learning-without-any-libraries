# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:46:29 2019

@author: purandur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Wine.csv')


X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



mean=np.mean(X_train,axis=0)

Cx=np.matmul((X_train-mean).T,(X_train-mean))



Cx=np.cov((X_train-mean).T)




        

#Sb=n*np.dot((MeanMatrix-MeanOfMeans).T,(MeanMatrix-MeanOfMeans))
#Sb=n*np.cov(MeanMatrix.T)    
#computee sw^-1Sb
eig_vals,eig_vecs=np.linalg.eig(Cx)
eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[i]) for i in range(len(eig_vals))]
eig_pairs=sorted(eig_pairs,key=lambda k:k[0],reverse=True)
W=np.hstack((eig_pairs[0][1].reshape(13,1),eig_pairs[1][1].reshape(13,1)))

X_new=np.dot(X_train,W)


clr=['red','blue','green']
for i in range(X_new.shape[0]):
    plt.scatter(X_new[i,0],X_new[i,1],c=clr[y_train[i]-1])
plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_new, y_train)

X_test_new=np.dot(X_test,W)
# Predicting the Test set results
y_pred = classifier.predict(X_test_new)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_new1 = lda.fit_transform(X_train, y_train)
for i in range(X_new1.shape[0]):
    plt.scatter(X_new1[i,0],X_new1[i,1],c=clr[y_train[i]-1])
plt.show()
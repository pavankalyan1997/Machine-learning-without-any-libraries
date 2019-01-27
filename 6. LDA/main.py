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



ClassDict={}
for i in range(y_train.shape[0]):
    if y_train[i] not in ClassDict:
        ClassDict[y_train[i]]=X_train[i].reshape(1,X_train[i].shape[0])
    else:
        ClassDict[y_train[i]]=np.append(ClassDict[y_train[i]],X_train[i].reshape(1,X_train[i].shape[0]),axis=0)

MeanDict={}
for each in ClassDict:
    MeanDict[each]=np.mean(ClassDict[each],axis=0)

#compute With in class scatter matrix
n=len(ClassDict)
Sw=np.array([])
for i in ClassDict:
    x=ClassDict[i]
    m=MeanDict[i]
    temp=np.dot((x-m).T,(x-m))
    if Sw.shape[0]==0:
        Sw=np.cov((x-m).T)
        #Sw=temp
    else:
        #SW=Sw+temp
        SW=Sw+np.cov((x-m).T)


#computer between class Scatter matrix
Sb=np.array([])
Mean=np.mean(X_train,axis=0)
Mean=Mean.reshape(Mean.shape[0],1)

for i in MeanDict:
    m=MeanDict[i].reshape(MeanDict[i].shape[0],1)
    n=len(ClassDict[i])
    temp=np.multiply(n,np.dot((m-Mean),(m-Mean).T))
    if Sb.shape[0]==0:
        Sb=temp
    else:
        Sb=Sb+temp
        

#Sb=n*np.dot((MeanMatrix-MeanOfMeans).T,(MeanMatrix-MeanOfMeans))
#Sb=n*np.cov(MeanMatrix.T)    
#computee sw^-1Sb
eig_vals,eig_vecs=np.linalg.eigh(np.matmul(np.linalg.pinv(Sw),Sb))
eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[i]) for i in range(len(eig_vals))]
eig_pairs=sorted(eig_pairs,key=lambda k:k[0],reverse=True)
W=np.hstack((eig_pairs[0][1].reshape(13,1),eig_pairs[1][1].reshape(13,1)))

X_new=np.dot(X_train,W)


clr=['red','blue','green']
for i in range(X_new.shape[0]):
    plt.scatter(X_new[i,0],X_new[i,1],c=clr[y_train[i]-1])
plt.show()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_new1 = lda.fit_transform(X_train, y_train)
for i in range(X_new1.shape[0]):
    plt.scatter(X_new1[i,0],X_new1[i,1],c=clr[y_train[i]-1])
plt.show()
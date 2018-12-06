# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 23:31:45 2018

@author: purandur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset=pd.read_csv('Social_Network_Ads.csv')

#split the dataset into training and testing set
dataset=dataset.sample(frac=1)
testset_size=int(dataset.shape[0]*0.25)
trainset_size=dataset.shape[0]-testset_size


#training data
X_train=dataset.iloc[0:trainset_size,2:4].values
y_train=dataset.iloc[0:trainset_size,4].values
#testing data
X_test=dataset.iloc[trainset_size:,2:4].values
y_test=dataset.iloc[trainset_size:,4].values




"""
P(class/x)=p(f1/class)*p(f2/class)*p(f3/class)*.....p(fn/class) *p(class) where fi= ith feature in database
p(fi/class) is modelled as guassian distribution:
    p(fi/class)=1/(sqrt(2*pi)*sigma)* e^-(testset's ith feature-avg(trainset's ith feature))**2/(2*sigma**2)
"""
#as it is a two class problem
class_data_dic={}
class_data_dic[0]=np.array([[]])
class_data_dic[1]=np.array([[]])
first_one=True
first_zero=True

for i in range(y_train.shape[0]):
    X_temp=X_train[i,:].reshape(X_train[i,:].shape[0],1)
    if y_train[i]==1:
        if first_one==True:
            class_data_dic[1]=X_temp
            first_one=False
        else:
            class_data_dic[1]=np.append(class_data_dic[1],X_temp,axis=1)
    elif y_train[i]==0:
        if first_zero==True:
            class_data_dic[0]=X_temp
            first_zero=False
        else:
            class_data_dic[0]=np.append(class_data_dic[0],X_temp,axis=1)
        
class_data_dic[0]=class_data_dic[0].T
class_data_dic[1]=class_data_dic[1].T
class0_count=class_data_dic[0].shape[0]
class1_count=class_data_dic[1].shape[0]
total_count=X_train.shape[0]

def likelyhood(x,mean,sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

    
X_train_mean_0=np.mean(class_data_dic[0],axis=0)
X_train_mean_1=np.mean(class_data_dic[1],axis=0)

X_train_std_0=np.std(class_data_dic[0],axis=0)
X_train_std_1=np.std(class_data_dic[1],axis=0)
def posterior(X,X_train_class,mean,std):
    product=np.prod(likelyhood(X,mean,std),axis=1)
    product=product*(X_train_class.shape[0]/X_train.shape[0])
    return product
    
    
p_1=posterior(X_test,class_data_dic[1],X_train_mean_1,X_train_std_1)
p_0=posterior(X_test,class_data_dic[0],X_train_mean_0,X_train_std_0)




y_pred=1*(p_1>p_0)

    
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)











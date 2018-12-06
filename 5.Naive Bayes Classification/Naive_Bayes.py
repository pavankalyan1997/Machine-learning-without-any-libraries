# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:20:34 2018

@author: purandur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Naive_Bayes():
    def __init__(self):
        self.class_data_dic={}
    
    
        
    def fit(self,X_train,y_train):
        def generate_data(class_data_dic,X_train,y_train):
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
            return class_data_dic
    
        self.X_train=X_train
        self.y_train=y_train
        self.class_data_dic[0]=np.array([[]])
        self.class_data_dic[1]=np.array([[]])
        self.class_data_dic=generate_data(self.class_data_dic,self.X_train,self.y_train)
        self.class_data_dic[0]=self.class_data_dic[0].T
        self.class_data_dic[1]=self.class_data_dic[1].T
        self.mean_0=np.mean(self.class_data_dic[0],axis=0)
        self.mean_1=np.mean(self.class_data_dic[1],axis=0)
        self.std_0=np.std(self.class_data_dic[0],axis=0)
        self.std_1=np.std(self.class_data_dic[1],axis=0)
        
        
    def predict(self,X_test):
        def posterior(X,X_train_class,mean_,std_):
            def likelyhood(x,mean,sigma):
                return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))
            
            product=np.prod(likelyhood(X,mean_,std_),axis=1)
            product=product*(X_train_class.shape[0]/self.X_train.shape[0])
            return product
        
        p_1=posterior(X_test,self.class_data_dic[1],self.mean_1,self.std_1)
        p_0=posterior(X_test,self.class_data_dic[0],self.mean_0,self.std_0)
        return 1*(p_1>p_0)
        
        
    
    
    
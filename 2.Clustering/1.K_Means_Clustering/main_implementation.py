# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:08:43 2018

@author: purandur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm

dataset=pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
m=X.shape[0]
n_iter=100


from Kmeans import Kmeans
f,axarr=plt.subplots(5,2,figsize=(15,30))
i=0
j=0
WCSS_array=np.array([])
for K in range(1,11):
    kmeans=Kmeans(X,K)
    kmeans.fit(n_iter)
    Output,Centroids=kmeans.predict()
    wcss=0
    for k in range(K):
        wcss+=np.sum((Output[k+1]-Centroids[k,:])**2)
    WCSS_array=np.append(WCSS_array,wcss)
    for k in range(K):
        axarr[i,j].scatter(Output[k+1][:,0],Output[k+1][:,1])
    axarr[i,j].scatter(Centroids[:,0],Centroids[:,1],s=300,c='yellow',label='Centroids')
    axarr[i,j].set_title('Clustered data with '+str(K)+' clusters')
    if(K%2==1):
        j+=1
    else:
        j=0
        i+=1
for ax in axarr.flat:
    ax.set(xlabel='Income', ylabel='Number of transactions')
    
    #WCSS_array=np.append(WCSS_array,kmeans.WCSS())
    
K_array=np.arange(1,11,1)
plt.plot(K_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()

K=5



kmeans=Kmeans(X,K)
kmeans.fit(n_iter)
Output,Centroids=kmeans.predict()

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[:,0],Centroids[:,1],s=300,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()



"""
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), WCSS_array)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
"""
#random initialization
i=rd.randint(0,X.shape[0])
Centroid_temp=np.array([X[i]])
K=5
for k in range(1,K):
    D=np.array([]) 
    for x in X:
        D=np.append(D,np.min(np.sum((x-Centroid_temp)**2)))
    prob=D/np.sum(D)
    cummulative_prob=np.cumsum(prob)
    r=rd.random()
    i=0
    for j,p in enumerate(cummulative_prob):
        if r<p:
            i=j
            break
    Centroid_temp=np.append(Centroid_temp,[X[i]],axis=0)
    
    
Centroids_rand=np.array([]).reshape(2,0)


for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids_rand=np.c_[Centroids_rand,X[rand]]
plt.scatter(X[:,0],X[:,1])
plt.scatter(Centroid_temp[:,0],Centroid_temp[:,1],s=200,color='yellow',label='Kmeans++')
plt.scatter(Centroids_rand[0,:],Centroids_rand[1,:],s=200,color='black',label='Random')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()
        
    
    
   







        




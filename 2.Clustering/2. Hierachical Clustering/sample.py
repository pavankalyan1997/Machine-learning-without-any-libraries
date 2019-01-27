# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 00:18:43 2018

@author: purandur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist 

X=np.array([[1,4],[3,2],[5,6],[3,4],[10,8],[11,9],[13,10],[15,9],[13,9],[5,5],[10,2],[11,4],[12,3],[13,5]]).astype(float)
plt.scatter(X[:,0],X[:,1],c='black')
plt.show()


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

plt.scatter(X[:,0],X[:,1],c='black')
plt.show()


Cluster_Centers=X.copy()
Clusters=np.array([i for i in range(Cluster_Centers.shape[0])]).reshape(Cluster_Centers.shape[0],1)
Clusters={}
for i in range(X.shape[0]):
    Clusters[i]=X[i]

#Centers=np.delete(Centers,0,axis=0)
HistoryCentroid=np.array([])
HistoryED=np.array([])
Cluster_Centers=Cluster_Centers[np.argsort(np.sum(Cluster_Centers**2,axis=1))]

while Cluster_Centers.shape[0]!=1:
    dist=np.tril(distance_matrix(Cluster_Centers,Cluster_Centers))
    indices=np.where(dist==np.min(dist[dist!=0]))[0]
    if type(indices)==np.ndarray:
        i1=indices[0]
        i2=np.where(dist==np.min(dist[dist!=0]))[1][0]
    else:
        i1=int(np.where(dist==np.min(dist[dist!=0]))[0])
        i2=int(np.where(dist==np.min(dist[dist!=0]))[1])
        
    if i1>i2:
        i1,i2=i2,i1
        
    ed=np.min(dist[dist!=0])
    c1=0
    c2=0
    #Clusters[Cluster_Centers.tolist().index(Cluster_Centers[i2].tolist())]=Clusters[Cluster_Centers.tolist().index(Cluster_Centers[i1].tolist())]
    if Cluster_Centers[i1].tolist() in HistoryCentroid.tolist():
        pos=HistoryCentroid.tolist().index(Cluster_Centers[i1].tolist())
        c1=HistoryED[pos]
        
    if Cluster_Centers[i2].tolist() in HistoryCentroid.tolist():
        pos=HistoryCentroid.tolist().index(Cluster_Centers[i2].tolist())
        c2=HistoryED[pos]
    
    if HistoryCentroid.size==0:
        HistoryCentroid=np.hstack((HistoryCentroid,(Cluster_Centers[i1]+Cluster_Centers[i2])/(2)))
        HistoryED=np.hstack((HistoryED,ed))
    else:
        HistoryCentroid=np.vstack((HistoryCentroid,(Cluster_Centers[i1]+Cluster_Centers[i2])/(2)))
        HistoryED=np.vstack((HistoryED,ed))
    
    
    
    
    #edDict[i1]=ed
    p1=np.sqrt(np.sum(Cluster_Centers[i1]**2))
    p2=np.sqrt(np.sum(Cluster_Centers[i2]**2))
    
    Cluster_Centers[i1]=(Cluster_Centers[i1]+Cluster_Centers[i2])/(2)
    Cluster_Centers=np.delete(Cluster_Centers,i2,axis=0)
    
    
    plt.plot([p1,p1,p2,p2],[c1,ed,ed,c2])
    plt.ylim(0,100)
    

plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
 
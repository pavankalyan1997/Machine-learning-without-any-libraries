# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:15:56 2019

@author: purandur
"""

import numpy as np
#generate sample data for classification
x1=np.random.rand(100)
x2=np.random.rand(100)
x3=np.random.rand(100)
x4=np.random.rand(100)
X=np.array([x1,x2,x3,x4])
y=np.sin(x1+x2+x3+x4).reshape(1,100)
"""for i in range(len(y)):
    if y[i]>0.8:y[i]=1
    else: y[i]=0"""

n_hidden_layers=1
n_neurons=X.shape[0]+1
theta1=np.random.rand(X.shape[0]*n_neurons).reshape(X.shape[0],n_neurons)
b1=np.random.randn(n_neurons).reshape(n_neurons,1)

theta2=np.random.rand(n_neurons*1).reshape(n_neurons,1)
b2=np.random.rand(1)

for i in range(100000):
    z=theta1.T.dot(X)+b1
    h=1/(1+np.exp(-z))
    
    ycap=theta2.T.dot(h)+b2
    
    dtheta2=h.dot((ycap-y).T)
    db2=np.sum(ycap-y)
    
    dtheta1=X.dot(((h*(1-h))*theta2.dot((ycap-y))).T)
    db1=np.sum(((h*(1-h))*theta2.dot((ycap-y))),axis=1).reshape(b1.shape[0],1)
    
    theta1=theta1-(0.0001)*dtheta1
    b1=b1-(0.0001)*db1
    theta2=theta2-(0.0001)*dtheta2
    b2=b2-(0.0001)*db2

z=theta1.T.dot(X)+b1
h=1/(1+np.exp(-z))
ycap=theta2.T.dot(h)+b2

y=y.T
ycap=ycap.T
for i in range(y.shape[0]):
    if y[i]>0.8:y[i]=1
    else: y[i]=0
    
for i in range(ycap.shape[0]):
    if ycap[i]>0.8:ycap[i]=1
    else: ycap[i]=0
    
print(np.sum(1*(y==ycap)))

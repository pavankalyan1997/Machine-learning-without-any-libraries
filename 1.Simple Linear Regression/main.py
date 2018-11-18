import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LinearRegression import LinearRegression

Data=pd.read_csv('Salary_Data.csv')
print(Data)
Data.describe()


#plot of dataset
plt.scatter(Data.iloc[:,0:1].values,Data.iloc[:,1].values)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of Data set')
plt.show()

#training and testing size
train_size=int(0.7*Data.shape[0])
test_size=int(0.3*Data.shape[0])
print("Training set size : "+ str(train_size))
print("Testing set size : "+str(test_size))


#training set split
#shuffle the dataset
Data=Data.sample(frac=1)
X=Data.iloc[:,0:1].values
y=Data.iloc[:,1].values

from FeatureScaling import FeatureScaling
fs=FeatureScaling(X,y)
X=fs.fit_transform_X()
y=fs.fit_transform_Y()


#training set split
X_train=X[0:train_size,:]
Y_train=y[0:train_size]

print(X_train.shape)
print(Y_train.shape)

#testing set split
X_test=X[train_size:,:]
Y_test=y[train_size:]




print(X_test.shape)
print(Y_test.shape)


#scatter plot of training set
plt.scatter(X_train,Y_train)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of training set')
plt.show()

#scatter plot of testing set
plt.scatter(X_test,Y_test)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of testing set')
plt.show()




lr=LinearRegression(X_train,Y_train)

theta=lr.returnTheta()
print(theta)

#testing set prediction
y_pred_normal,error_percentage=lr.predictUsingNormalEquation(X_test,Y_test)
y_pred_normal=fs.inverse_transform_Y(y_pred_normal)
print(error_percentage)

#training set prediction
y_pred_train_normal,error_percentage_train_normal=lr.predictUsingNormalEquation(X_train,Y_train)
y_pred_train_normal=fs.inverse_transform_Y(y_pred_train_normal)
print(lr.computeCostFunction())


#learning parameters
n_iter=1000
alpha=0.05

theta,J_Array,theta_array=lr.performGradientDescent(n_iter,alpha)


y_pred_grad,ErrorPercentage=lr.predict(X_test,Y_test)
print(ErrorPercentage)
y_pred_grad=fs.inverse_transform_Y(y_pred_grad)

#let's see how train set is predicted
y_pred_train,error_for_train=lr.predict(X_train,Y_train)
y_pred_train=fs.inverse_transform_Y(y_pred_train)
print(error_for_train)

#inverse scaling the features
X_train=fs.inverse_transform_X(X_train)
Y_train=fs.inverse_transform_Y(Y_train)
X_test=fs.inverse_transform_X(X_test)
Y_test=fs.inverse_transform_Y(Y_test)

#let's see how train set is predicted using gradient descent
plt.scatter(X_train,Y_train)
plt.plot(X_train,y_pred_train,'r')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Training set prediction using Gradient Descent')
plt.show()


#let's see how train set is predicted using normal equation
plt.scatter(X_train,Y_train)
plt.plot(X_train,y_pred_train_normal,'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Training set prediction using Normal Equation')
plt.show()


#let's see how test set is predicted using gradient descent
plt.scatter(X_test,Y_test)
plt.plot(X_test,y_pred_grad,'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Test set prediction using Gradient Descent')
plt.show()

#let's see how test set is predicted using normal equation method
plt.scatter(X_test,Y_test)
plt.plot(X_test,y_pred_normal,'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Test set prediction using Normal Equation')
plt.show()

#plot of how cost function is minimized as number of iterations is proceeded
x=[i for i in range(1000)]
plt.plot(x,J_Array)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function(J)')
plt.show()
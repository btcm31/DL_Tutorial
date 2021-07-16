# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:06:34 2019

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
X = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)
""" plt.plot(X,y)
plt.xlabel('Dien tich')
plt.ylabel('Gia')
plt.show() """

X = np.hstack((np.ones((N,1)),X))

numOfIter = 100
cost = np.zeros((3,1))
lr_lst = [0.00001]
y_pred = np.zeros((3,N))
w = ""
for idx,lr in enumerate(lr_lst):
    w = np.array([0.,1.]).reshape(-1,1)
    for i in range(1,numOfIter):
        r = np.dot(X,w) - y #np.dot  => dot((M*N),(N*R)) = array(M*R)\
        w[0] = w[0] - lr * np.sum(r)
        w[1] = w[1] - lr * np.sum(np.multiply(r,X[:,1].reshape(-1,1)))
    a = np.dot(X,w)-y
    cost[idx] += 1/2 * np.sum(a*a)# * operation carries out element-wise multiplication on array elements.
    y_pred[idx] = np.dot(X,w).reshape(1,-1)

plt.plot(y_pred[0],'--')
plt.plot(y,'+')
plt.legend()
plt.show()
print(cost)
print(w)


x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)
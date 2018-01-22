#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:46:03 2017

@author: zhaoqingyang
"""
#Code for HW02-5 b

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat_hw2_a = sio.loadmat('/Users/zhaoqingyang/Desktop/Fall17/289A/Homework/HW02/hw02-data/1D_poly.mat')

# Preparing for training data 
x_train = np.array(mat_hw2_a['x_train'])
y_train = np.array(mat_hw2_a['y_train'])

# polyFit is a function which used to fit Peach Flight Time with regard to Degree D
def featureMatrix( D, x_train):
    temp = np.ones(len(x_train))
    for i in range(1, D + 1):
        temp = np.append(temp,np.power(x_train,i))
    temp.shape = (D + 1,len(x_train))
    temp = np.transpose(temp)
    return temp
    
def polyFit( D, x_train, y_train ):
    temp = featureMatrix(D, x_train)
    para = np.dot(temp.T, temp)
    para2 = np.dot(temp.T, y_train)
    w = np.linalg.solve(para, para2)
#    para = np.linalg.inv(np.dot(np.transpose(temp),temp))
#    para = np.dot(para,np.transpose(temp))
#    para = np.dot(para,y_train)
    #return the training error
    return w
   
D = len(x_train) - 1
trainingError = np.zeros(D)
for i in range(1, D + 1):
    para = polyFit( i, x_train, y_train)
    trainingError[i-1] = np.linalg.norm(np.dot(featureMatrix( i, x_train), para) - y_train,2) ** 2 / len(x_train)

plt.scatter(range(1, D + 1),trainingError)

#Code for HW02-5 d


freshError = np.zeros(D)
y_fresh = np.array(mat_hw2_a['y_fresh'])
for i in range(1,D + 1):
    para = polyFit( i, x_train, y_train)
    freshError[i-1] = np.linalg.norm(np.dot(featureMatrix( i, x_train), para) - y_fresh,2) ** 2 / len(x_train)
    
plt.scatter(range(1,D+1),freshError)


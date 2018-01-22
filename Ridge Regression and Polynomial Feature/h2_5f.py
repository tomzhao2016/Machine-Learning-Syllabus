#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:53:49 2017

@author: zhaoqingyang
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import operator as op

mat_hw2_f = sio.loadmat('/Users/zhaoqingyang/Desktop/Fall17/289A/Homework/HW02/hw02-data/polynomial_regression_samples.mat')

x = np.array(mat_hw2_f['x'])
y = np.array(mat_hw2_f['y'])
# D equals 1 to 4
def featureMatrix( D, x ):
    k = 0
    if D == 1:
        fMat = np.zeros((len(x),6),dtype = np.float)
        perm = list(set(permutations([1, 0, 0, 0, 0, 0])))   
        for i in perm:
            for j in range(len(x)):
                fMat[j,k] = i[0] + x[j,0]**i[1] + x[j,1]**i[2] + x[j,2]**i[3] + x[j,3]**i[4] + x[j,4]**i[5]
            k = k + 1
    if D == 2:
        fMat = np.zeros((len(x),21),dtype = np.float)
        perm = list(set(permutations([1, 1, 0, 0, 0, 0])))
        perm = perm + list(set(permutations([2, 0, 0, 0, 0, 0])))    
        for i in perm:
            for j in range(len(x)):
                fMat[j,k] = i[0] + x[j,0]**i[1] + x[j,1]**i[2] + x[j,2]**i[3] + x[j,3]**i[4] + x[j,4]**i[5]
            k = k + 1
    if D == 3:
        fMat = np.zeros((len(x),56),dtype = np.float)
        perm = list(set(permutations([1, 1, 1, 0, 0, 0])))
        perm = perm + list(set(permutations([1, 2, 0, 0, 0, 0])))
        perm = perm + list(set(permutations([3, 0, 0, 0, 0, 0])))
        for i in perm:
            for j in range(len(x)):
                fMat[j,k] = i[0] + x[j,0]**i[1] + x[j,1]**i[2] + x[j,2]**i[3] + x[j,3]**i[4] + x[j,4]**i[5]
            k = k + 1
    if D == 4:
        fMat = np.zeros((len(x),126),dtype = np.float)
        perm = list(set(permutations([1, 1, 1, 1, 0, 0])))
        perm = perm + list(set(permutations([1, 1, 2, 0, 0, 0])))
        perm = perm + list(set(permutations([1, 3, 0, 0, 0, 0])))
        perm = perm + list(set(permutations([4, 0, 0, 0, 0, 0])))
        perm = perm + list(set(permutations([2, 2, 0, 0, 0, 0])))       
        for i in perm:
            for j in range(len(x)):
                fMat[j,k] = i[0] + x[j,0]**i[1] + x[j,1]**i[2] + x[j,2]**i[3] + x[j,3]**i[4] + x[j,4]**i[5]
            k = k + 1
    
    return fMat

crossNum = np.int(len(x)/4 )
error = np.zeros((4,4),dtype = np.float)
trainingError = np.zeros((4,4),dtype = np.float)
for i in range(1,5):
    fMat = featureMatrix(i,x)
    for c in range(4):
        vtrain = np.concatenate((fMat[range(0,c*crossNum)],fMat[range((c+1)*crossNum,crossNum*4)]),axis = 0)
        vtest = fMat[range(c*crossNum,(c+1)*crossNum)]
        vtrainy = np.concatenate((y[range(0,c*crossNum)],y[range((c+1)*crossNum,crossNum*4)]),axis = 0)
        vtesty = y[range(c*crossNum,(c+1)*crossNum)]
        w = np.linalg.solve(np.dot(vtrain.T,vtrain),np.dot(vtrain.T,vtrainy))
        trainingError[c,i-1] = np.linalg.norm(np.dot(vtrain,w) - vtrainy) ** 2 / (crossNum*3) 
        error[c,i-1] = np.linalg.norm(np.dot(vtest,w) - vtesty) ** 2 / crossNum 



plt.scatter(range(1,5),np.sum(error,0))
plt.scatter(range(1,5),np.sum(trainingError,0))
# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Qingyang Zhao
Date: 8/31/2017
CS289A H01_6_f
"""

import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('/Users/zhaoqingyang/Desktop/Fall17/289A/Homework/HW01/hw01-data/system_identification_train.mat')
mat_test = sio.loadmat('/Users/zhaoqingyang/Desktop/Fall17/289A/Homework/HW01/hw01-data/system_identification_eval.mat')

mat_contents['xd'] = mat_contents['xd'] + 29.0576
mat_contents['xdp'] = mat_contents['xdp'] + 29.0576

mat_train = np.array([mat_contents['x'][0,],mat_contents['xd'][0,],mat_contents['xp'][0,],mat_contents['xdp'][0,]])

para = np.linalg.inv(np.dot(mat_train,np.transpose(mat_train)))
para = np.dot(para,mat_train)
para = np.dot(para,np.transpose(np.array(mat_contents['xdd'])))


x = np.zeros(150) 
x[0] = mat_test['x0'][0,][0]
x_dot = np.zeros(150) 
x_dot[0] = mat_test['xd0'][0,][0] +  29.0576
xp = mat_test['xp'][0,][1:150]
mat_test['xdp'] = mat_test['xdp'] + 29.0576
xp_dot = mat_test['xdp'][0,][1:150]
xdd = np.zeros(150)
for i in range(0,149):
    xdd[i] = np.dot(np.array([x[i],x_dot[i],xp[i],xp_dot[i]]),para)
    i = i + 1
    x[i] = x[i-1] + x_dot[i-1]*1/15
    x_dot[i] = x_dot[i-1] + xdd[i-1]*1/15

np.savetxt('submission.txt',x,newline='\n')

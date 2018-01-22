#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 01:31:43 2017
HW01-a
@author: Qingyang Zhao
"""

import scipy.io as sio
import numpy as np

mat_a = sio.loadmat('/Users/zhaoqingyang/Desktop/Fall17/289A/Homework/HW01/hw01-data/system_identification_programming_a.mat')


A = np.array([mat_a['x'][0,],mat_a['u'][0,]])
x_1 = np.array([mat_a['x'][0,1:30],])
A = A[:,0:29]

para = np.linalg.inv(np.dot(A,np.transpose(A)))
para = np.dot(para,A)
para = np.dot(para,np.transpose(x_1))


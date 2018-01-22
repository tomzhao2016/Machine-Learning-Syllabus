import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections
from utils import compute_covariance_matrix



class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		X = np.array(X)
		Y = np.array(Y)
		self.mu = []
		for i in range(self.NUM_CLASSES) :
			index = Y == i
			n = sum(index)
			Xi = X[index,:]
			mean = np.mean(Xi, axis = 0)
			self.mu.append(mean)
			Xi_minus_mean = Xi - mean
			if i == 0 :
				self.cov = n * compute_covariance_matrix(Xi_minus_mean, Xi_minus_mean)
			else :
				self.cov += n * compute_covariance_matrix(Xi_minus_mean, Xi_minus_mean)

		self.cov = self.cov / X.shape[0]
		
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		pred = np.zeros((self.NUM_CLASSES,1))

		for j in range(self.NUM_CLASSES) :
			pred[j] = -np.matmul(np.matmul((x - self.mu[j]).T, inv(self.cov)), x - self.mu[j])
		return np.argmax(pred)

	
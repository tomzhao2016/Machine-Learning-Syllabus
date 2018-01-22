
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections
from utils import compute_covariance_matrix


class QDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		X = np.array(X)
		Y = np.array(Y)
		self.mu = []
		self.cov = []

		for i in range(self.NUM_CLASSES) :
			index = Y == i
			n = sum(index)
			Xi = X[index, :]
			mean = np.mean(Xi, axis = 0)
			self.mu.append(mean)
			Xi_minus_mean = Xi - mean
			cov = compute_covariance_matrix(Xi_minus_mean, Xi_minus_mean)
			self.cov.append(cov)

		
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		pred = np.zeros((self.NUM_CLASSES, 1))

		for j in range(self.NUM_CLASSES) :
			pred[j] = -np.matmul(np.matmul((x - self.mu[j]).T, inv(self.cov[j])), x - self.mu[j]) - np.log(det(self.cov[j]))
		return np.argmax(pred)

	
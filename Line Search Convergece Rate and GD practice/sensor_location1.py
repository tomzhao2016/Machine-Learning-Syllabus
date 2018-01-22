import numpy as np
import scipy.spatial

########################################################################
#########  Data Generating Functions ###################################
########################################################################
def generate_sensors(k = 7, d = 2):
	"""
	Generate sensor locations. 
	Input:
	k: The number of sensors.
	d: The spatial dimension.
	Output:
	sensor_loc: k * d numpy array.
	"""
	sensor_loc = 100*np.random.randn(k,d)
	return sensor_loc

def generate_data(sensor_loc, k = 7, d = 2, 
				 n = 1, original_dist = True):
	"""
	Generate the locations of n points.  

	Input:
	sensor_loc: k * d numpy array. Location of sensor. 
	k: The number of sensors.
	d: The spatial dimension.
	n: The number of points.
	original_dist: Whether the data are generated from the original 
	distribution. 

	Output:
	obj_loc: n * d numpy array. The location of the n objects. 
	distance: n * k numpy array. The distance between object and 
	the k sensors. 
	"""
	assert k, d == sensor_loc.shape

	obj_loc = 100*np.random.randn(n, d)
	if not original_dist:
	   obj_loc += 1000
	   
	distance = scipy.spatial.distance.cdist(obj_loc, 
										   sensor_loc, 
										   metric='euclidean')
	distance += np.random.randn(n, k)  
	return obj_loc, distance
##################################################################
# Starter code for Part (b)
##################################################################
np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc)
single_distance = distance[0]


##################################################################
# Starter code for Part (c)
##################################################################
def generate_data_given_location(sensor_loc, obj_loc, k = 7, d = 2):
	"""
	Generate the distance measurements given location of a single object and sensor. 

	Input:
	obj_loc: 1 * d numpy array. Location of object
	sensor_loc: k * d numpy array. Location of sensor. 
	k: The number of sensors.
	d: The spatial dimension. 

	Output: 
	distance: 1 * k numpy array. The distance between object and 
	the k sensors. 
	"""
	assert k, d == sensor_loc.shape 
	 
	distance = scipy.spatial.distance.cdist(obj_loc, 
					   sensor_loc, 
					   metric='euclidean')
	distance += np.random.randn(1, k)  
	return obj_loc, distance

def log_likelihood(obj_loc, sensor_loc, distance): 
	"""
	This function computes the log likelihood (as expressed in Part a).
	Input: 
	obj_loc: shape [1,2]
	sensor_loc: shape [7,2]
	distance: shape [7]
	Output: 
	The log likelihood function value. 
	"""  
	diff_distance = np.sqrt(np.sum((sensor_loc - obj_loc)**2, axis = 1))- distance
	func_value = -sum((diff_distance)**2)/2
	return func_value


np.random.seed(100)
# Sensor locations. 
sensor_loc = generate_sensors()
num_gd_replicates = 100

# Object locations. 
obj_locs = [[[i,i]] for i in np.arange(100,1000,100)]  















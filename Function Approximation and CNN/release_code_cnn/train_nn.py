from data_manager import data_manager
from nn_classifier import NN
from trainer import Solver


import matplotlib.pyplot as plt




CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']


image_size = 90
classes = CLASS_LABELS

dm = data_manager(classes, image_size)

val_data = dm.val_data
train_data = dm.train_data

K = [1, 20, 100]
test_losses = []
train_losses = []

for k in K: 
	nn = NN(val_data,train_data,n_neighbors=k)

	nn.train_model()

	test_losses.append(nn.get_validation_error())
	train_losses.append(nn.get_train_error())




plt.plot(K, test_losses,label = 'Validation')
plt.plot(K, train_losses, label = 'Training')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Loss')
plt.show()


#####Plot the test error and training error###


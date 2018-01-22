from data_manager import data_manager
from cnn import CNN
from trainer import Solver
from viz_features import Viz_Feat
import random
import tensorflow as tf


import matplotlib.pyplot as plt

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

LITTLE_CLASS_LABELS = ['apple','banana','eggplant']

image_size = 90

random.seed(0)

classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

solver = Solver(cnn,dm)

solver.optimize()

plt.plot(solver.test_losses,label = 'Validation')
plt.plot(solver.train_losses, label = 'Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('Accuracy')
plt.show()

val_data = dm.val_data
train_data = dm.train_data

sess = solver.sess
#saver = tf.train.Saver()

#saver.save(sess, 'model_final')
#plt.show()
cm = Viz_Feat(val_data,train_data,CLASS_LABELS,sess)

cm.vizualize_features(cnn)





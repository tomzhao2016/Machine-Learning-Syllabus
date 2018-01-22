from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np



class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels,sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess





    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute the response map for the index images
        '''
        f, axarr = plt.subplots(len(images), 5)
        j = 0
        for img in images:
            val_feature = np.asarray([self.val_data[img]['features']], dtype = np.float32)
            val_label = np.asarray([self.val_data[img]['label']],dtype = np.float32)
            feed_dict = {net.images: val_feature, net.labels: val_label}
            out_img = self.sess.run(net.conv2d_out, feed_dict = feed_dict)
            # the shape o
            #f out_img is (1,90,90,5)
            for i in range(out_img.shape[3]):
                # cv2.namedWindow("Response Map")
                axarr[j, i].imshow(self.revert_image(out_img[0,:,:,i]))
                #plt.imshow(self.revert_image(out_img[0,:,:,i]))
            j = j + 1
        plt.show()



    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        





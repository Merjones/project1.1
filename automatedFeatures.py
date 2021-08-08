from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np


def getFeatures(img):
    x = np.zeros((1,224,224,3)) # size of image that VGG16 takes
    x[:,:,:,0] = img
    x[:,:,:,1] = img
    x[:,:,:,2] = img

    model = VGG16(weights='imagenet', include_top=False) #do not include the fully-connected layers w the softmax classifier
    #model = VGG16(weights='imagenet') #do not include the fully-connected layers w the softmax classifier
    imgData = preprocess_input(x)
    features = model.predict(imgData)
    features = features.reshape((1,7*7*512)) ## resize to be a 1D array that is the output of the last max pooling layer

    return features
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img

import numpy as np
import openImage
import numpy as np
import pandas as pd
import os
import PIL


# def getPatch(xMax, yMax, x_cent, y_cent):
#     top = y_cent - 112
#     bottom = y_cent + 112
#     left = x_cent - 112
#     right = x_cent + 112
#
#     if top < 0:
#         bottom += (top*-1)
#         top = 0
#     if bottom > yMax:
#         top -= (yMax-bottom)
#         bottom = yMax
#     if left < 0:
#         right += (left*-1)
#         left = 0
#     if right > xMax:
#         left -= (xMax-right)
#         right = xMax
#
#     im1 = img[left:right, top:bottom]
#     return im1
#
# imageFilePath = "C:\\Users\jones\Desktop\MammData\ccOnly\ARP0003_1001_RCC.img"
# massCenterMarks = pd.read_excel('onlyCC.xlsx')
#
# img = openImage.load_image(imageFilePath)
# #imgRaw, LorR = segmentImage.segment(imageFilePath, img)
# imageName = os.path.basename(imageFilePath)
#
# ## STEP 1 PREPROCESS
# #img = img.astype(np.uint8) ## convert from 12 bit to 8 bit
# massCenterMarks.set_index('image_name', inplace=True)
# x_center = massCenterMarks.loc[imageName]['X_Cent']
# y_center = massCenterMarks.loc[imageName]['Y_Cent']
# xMax, yMax = img.shape
# img1 = getPatch(xMax, yMax, x_center, y_center)

def getFeatures(img):
    x = np.zeros((1,224,224,3)) # size of image that VGG16 takes
    x[:,:,:,0] = img
    x[:,:,:,1] = img
    x[:,:,:,2] = img

    model = VGG16(weights='imagenet', include_top=False) #do not include the fully-connected layers w the softmax classifier
    imgData = preprocess_input(x)
    features = model.predict(imgData)
    features = features.reshape((1,7*7*512)) ## resize to be a 1D array that is the output of the last max pooling layer

    # model1 = VGG16(weights='imagenet')
    # imgData1 = preprocess_input(x)
    # features1 = model1.predict(imgData1)
    #features1 = features.reshape((1, 7 * 7 * 512))

    return features
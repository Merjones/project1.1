import numpy as np
import pandas as pd
import openImage
import handcraftedFeatures
import automatedFeatures
import os

## https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)


def getPatch(xMax, yMax, x_cent, y_cent):
    top = y_cent - 32
    bottom = y_cent + 32
    left = x_cent - 32
    right = x_cent + 32

    if top < 0:
        bottom += (top*-1)
        top = 0
    if bottom > yMax:
        top += (yMax-bottom)
        bottom = yMax
    if left < 0:
        right += (left*-1)
        left = 0
    if right > xMax:
        left += (xMax-right)
        right = xMax

    return int(left),int(right),int(top),int(bottom)

def features(dataFile, imagePath):
    handcraftedFeaturesDF = pd.DataFrame(
        columns=(list(range(0, 41))))  ##41 different handcrafted features plus the class label
    #handcraftedFeaturesDF.columns = [*handcraftedFeaturesDF.columns[:-1], 'ClassLabel']

    automatedFeaturesDF = pd.DataFrame(
        columns=(list(range(0, 1000))))  ##7x7x512 different handcrafted features plus the class label
    #automatedFeaturesDF.columns = [*automatedFeaturesDF.columns[:-1], 'ClassLabel']

    for index ,row in dataFile.iterrows():
        imageID, massType, xCent, yCent = row
        imageFilePath = os.path.join(imagePath, imageID)
        img_ = openImage.load_image(imageFilePath)
        yMax, xMax = img_.shape

        #get a 64x64 patch around the center of the lesion
        left ,right ,top ,bottom = getPatch(xMax, yMax, xCent, yCent)
        img = img_[top:bottom, left:right]

        # resize this to 224x224 since that is what VGG16 takes
        img = bilinear_resize_vectorized(img, 224, 224)
        print("bilinear resize success")

        ##to visualize the patches
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(img_)
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(img)
        # plt.show()

        ###get the handcrafted features
        # print("getting handcrafted")
        # handcrafted_features = handcraftedFeatures.getFeatures(img)
        # ## create a row that has the features and then class label as the last entry
        # # handcrafted_features.append(massType)
        # handcraftedFeaturesDF.loc[len(handcraftedFeaturesDF)] = handcrafted_features

        ###get the automated features
        print("getting automated")
        automated_features = automatedFeatures.getFeatures(img)
        # automated_features = np.append(automated_features, massType)
        automatedFeaturesDF.loc[len(automatedFeaturesDF)] = automated_features[0]
        print("got both")

    ##dump into a pickle
    # automatedFeaturesDF.to_pickle("allAutomatedFeatures.pkl")
    # handcraftedFeaturesDF.to_pickle("allHandcraftedFeatures.pkl")

    #return handcraftedFeaturesDF, automatedFeaturesDF
    return automatedFeaturesDF

#
# name = 'ARP0007_0901_RCC.img'
# (115, "ARP1052_1201_LCC.img")
# (753, "ARP0239_1001_LCC.img")
# (995,"FK0101000_1025_LCC.img")
#
# massCenterMarks = pd.read_excel('onlyCC.xlsx')
# massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0','Unnamed: 5', 'Unnamed: 6',
#                                                 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
#                                                 'Unnamed: 10'])
# imageFolderPath = "ccOnly"
# for index, row in massCenterMarks.iterrows():
#     imageID, massType, xCent, yCent = row
#     imageFilePath = os.path.join(imageFolderPath, imageID)
#     img_ = openImage.load_image(imageFilePath)
#     yMax, xMax = img_.shape
#     left, right, top, bottom = getPatchSize(xMax, yMax, xCent, yCent)
#     img = img_[top:bottom, left:right]
#     img = bilinear_resize_vectorized(img, 224, 244)
#     h = handcraftedFeatures.getFeatures(img)
#     if h[0] == 0:
#         print("This image is trash.")
#         print(row)

#
# imageID, massType, xCent, yCent = ["FK0101000_1025_LCC.img", 1, 482, 300] #955
# #imageID, massType, xCent, yCent = ["ARP0239_1001_LCC.img", 1, 196, 136] #753
# #imageID, massType, xCent, yCent = ["ARP1052_1201_LCC.img", 1, 530, 378] #115
#
# import matplotlib.pyplot as plt
#
# imageFilePath = os.path.join(imageFolderPath, imageID)
# img_ = openImage.load_image(imageFilePath)
# yMax, xMax = img_.shape
# left, right, top, bottom = getPatchSize(xMax, yMax, xCent, yCent)
# img = img_[top:bottom, left:right]
# img = bilinear_resize_vectorized(img,224, 244)
# img = img.astype(np.uint8)





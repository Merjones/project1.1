import numpy as np
import openImage
import handcraftedFeatures
import automatedFeatures
import pandas as pd
import os
from sklearn.model_selection import KFold


imageFolderPath = "ccOnly"
massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0','Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])
#massCenterMarks.set_index('image_name', inplace=True)
imageNames = massCenterMarks['image_name']
## mapping all malignancies to a 1 and all benign to a 3
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map({1:1, 2:1, 3:3, 5:3})
#multipleMasses = pd.read_excel('duplicates.xlsx') #will be used to make sure we take all mass center marks

def getPatchSize(xMax, yMax, x_cent, y_cent):
    top = y_cent - 112
    bottom = y_cent + 112
    left = x_cent - 112
    right = x_cent + 112

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
        left -= (xMax-right)
        right = xMax

    return int(left),int(right),int(top),int(bottom)

allImageNames = np.array(os.listdir(imageFolderPath))
cv = KFold(n_splits=10,random_state=1, shuffle=True)

count = 0
### WILL ACTUALLY LOOP THROUGH THIS EACH TIME FOR THE 10 ITERATIONS WHEN READY
for train,test in cv.split(imageNames):
    x_trainNames, x_testNames = imageNames[train], imageNames[test]
    y_train  = pd.Series()
    y_test = pd.Series()
    xTrainData = pd.DataFrame()
    xTestData = pd.DataFrame()
    for name in x_trainNames:
        # y_train = y_train.append(massCenterMarks.loc[massCenterMarks['image_name'] == name]['Mass Type'])
        xTrainData = xTrainData.append(massCenterMarks.loc[massCenterMarks['image_name'] == name])
    for name in x_testNames:
        # y_test = y_test.append(massCenterMarks.loc[massCenterMarks['image_name'] == name]['Mass Type'])
        xTestData = xTestData.append(massCenterMarks.loc[massCenterMarks['image_name'] == name])

    print('len of train: %s, len of test: %s' % (len(imageNames[train]), len(imageNames[test])))

    ## get the
    for index,row in xTrainData.iterrows():
        imageID = row['image_name']
        xCent = row['X_Cent']
        yCent = row['Y_Cent']

        imageFilePath = os.path.join(imageFolderPath, imageID)
        img_ = openImage.load_image(imageFilePath)
        xMax, yMax = img_.shape
        left,right,top,bottom = getPatchSize(xMax, yMax, xCent, yCent)
        img = img_[left:right, top:bottom]
        img = img.astype(np.uint8)  ## convert from 12 bit to 8 bit
    #

# testSet = pd.DataFrame()
# trainingSet = pd.DataFrame()
# for name in allImageNames[test]:
#     data = massCenterMarks.loc[name]
#     testSet = testSet.append(data)
# for name in allImageNames[train]:
#     data = massCenterMarks.loc[name]
#     trainingSet = trainingSet.append(data)

# testLabel = testSet["Mass Type"]
# trainingLabel = trainingSet['Mass Type']
#

#
#
# handcraftedFeaturesDF_train = pd.DataFrame(columns=(list(range(0,39)))) ##45 different handcrafted features plus the class label
# handcraftedFeaturesDF_train.columns = [*handcraftedFeaturesDF_train.columns[:-1], 'ClassLabel']
#
# automatedFeaturesDF_train = pd.DataFrame(columns=(list(range(0,25089)))) ## d7x7x512 different handcrafted features plus the class label
# automatedFeaturesDF_train.columns = [*automatedFeaturesDF_train.columns[:-1], 'ClassLabel']
#
# handcraftedFeaturesDF_test = pd.DataFrame(columns=(list(range(0,39)))) ##45 different handcrafted features plus the class label
# handcraftedFeaturesDF_test.columns = [*handcraftedFeaturesDF_test.columns[:-1], 'ClassLabel']
#
# automatedFeaturesDF_test = pd.DataFrame(columns=(list(range(0,25089)))) ## d7x7x512 different handcrafted features plus the class label
# automatedFeaturesDF_test.columns = [*automatedFeaturesDF_test.columns[:-1], 'ClassLabel']
#
# ##NEED TO DO SOMETHING TO MAKESURE YOU GET AL DUPLICATES
# for row in trainingSet.iterrows():
#     imageID, [massType, xCent, yCent] = row
#
#     ###get the image patch
#     imageFilePath = os.path.join(imageFolderPath, imageID)
#     img_ = openImage.load_image(imageFilePath)
#     xMax, yMax = img_.shape
#     left,right,top,bottom = getPatchSize(xMax, yMax, xCent, yCent)
#     img = img_[left:right, top:bottom]
#     img = img.astype(np.uint8)  ## convert from 12 bit to 8 bit
#
#     ###get the handcrafted features
#     print("getting handcrafted")
#     handcrafted_features = handcraftedFeatures.getFeatures(img)
#     ## create a row that has the features and then class label as the last entry
#     handcrafted_features.append(massType)
#     handcraftedFeaturesDF_train.loc[len(handcraftedFeaturesDF_train)] = handcrafted_features
#
#     ###get the automated features
#     print("getting automated")
#     automated_features = automatedFeatures.getFeatures(img)
#     automated_features = np.append(automated_features, massType)
#     automatedFeaturesDF_train.loc[len(automatedFeaturesDF_train)] = automated_features
#     print("got both")
#
# print("getting test set")
# for row in testSet.iterrows():
#     imageID, [massType, xCent, yCent] = row
#
#     ###get the image patch
#     imageFilePath = os.path.join(imageFolderPath, imageID)
#     img_ = openImage.load_image(imageFilePath)
#     xMax, yMax = img_.shape
#     left,right,top,bottom = getPatchSize(xMax, yMax, xCent, yCent)
#     img = img_[left:right, top:bottom]
#     img = img.astype(np.uint8)  ## convert from 12 bit to 8 bit
#
#     ###get the handcrafted features
#     print("getting handcrafted")
#     handcrafted_features = handcraftedFeatures.getFeatures(img)
#     ## create a row that has the features and then class label as the last entry
#     handcrafted_features.append(massType)
#     handcraftedFeaturesDF_test.loc[len(handcraftedFeaturesDF_test)] = handcrafted_features
#
#     ###get the automated features
#     print("getting automated")
#     automated_features = automatedFeatures.getFeatures(img)
#     automated_features = np.append(automated_features, massType)
#     automatedFeaturesDF_test.loc[len(automatedFeaturesDF_test)] = automated_features
#     print("got both")
#
# automatedFeaturesDF_test.to_pickle("autoTest.pkl")
# automatedFeaturesDF_train.to_pickle("autoTrain.pkl")
# handcraftedFeaturesDF_test.to_pickle("handTest.pkl")
# handcraftedFeaturesDF_train.to_pickle("handTrain.pkl")
#
#
#
# # # imageFilePath = "C:\\Users\jones\Desktop\MammData\ccOnly\ARP0003_1001_RCC.img"
# # # massCenterMarks = pd.read_excel('onlyCC.xlsx')
# #
# # img = openImage.load_image(imageFilePath)
# # #imgRaw, LorR = segmentImage.segment(imageFilePath, img)
# # imageName = os.path.basename(imageFilePath)
# #
# # ## STEP 1 PREPROCESS
# # img = img.astype(np.uint8) ## convert from 12 bit to 8 bit
#
# ## STEP 2 SPLIT INTO TRAINING AND TESTING
# ## STEP 3 GET HANDCRAFTED FEATURES
# #handcrafted_features = handcraftedFeatures.getFeatures(img)
#
#
# #automated_features = automatedFeatures
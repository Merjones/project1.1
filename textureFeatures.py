from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import openImage
import segmentImage
import glrlm
import math
import numpy as np
import matplotlib.pyplot as plt
from radiomics import cMatrices

# filePath = "C:\\Users\jones\Desktop\MammData\Test-Images-GE-format\M2000210_1612_LCC.img"
# imgRaw = openImage.load_image(filePath)
# img1 = imgRaw.astype(np.uint8)
# img, lorr = segmentImage.segmentImage(filePath, img1)

def getGLCMFeatures(img):
    img = img.astype(np.uint8)  ## convert from 12 bit to 8 bit
    glcm  = greycomatrix(img, distances=[1], angles=[0, (45*math.pi/180), (90*math.pi/180), (135*math.pi/180)])
    contrastMax = greycoprops(glcm, prop='contrast').max()
    contrastMean = greycoprops(glcm, prop='contrast').mean()
    dissimilarityMax = greycoprops(glcm, prop='dissimilarity').max()
    dissimilarityMean = greycoprops(glcm, prop='dissimilarity').mean()
    homogeneityMax = greycoprops(glcm, prop='homogeneity').max()
    homogeneityMean = greycoprops(glcm, prop='homogeneity').mean()
    ASMmax = greycoprops(glcm, prop='ASM').max()
    ASMmean = greycoprops(glcm, prop='ASM').mean()
    energyMax = greycoprops(glcm, prop='energy').max()
    energyMean = greycoprops(glcm, prop='energy').mean()
    correlationMax = greycoprops(glcm, prop='correlation').max()
    correlationMean = greycoprops(glcm, prop='correlation').mean()

    maxGLCMFeatures = [contrastMax, dissimilarityMax, homogeneityMax, ASMmax, energyMax, correlationMax]
    meanGLCMFeatures = [contrastMean, dissimilarityMean, homogeneityMean, ASMmean, energyMean, correlationMean]
    GLCMFeatures = [contrastMax, dissimilarityMax, homogeneityMax, ASMmax, energyMax, correlationMax,
                    contrastMean, dissimilarityMean, homogeneityMean, ASMmean, energyMean, correlationMean]

    return GLCMFeatures

def getLBPFeatures(img):
    lbp = local_binary_pattern(img,8,1,'uniform')  ## looking at the 3x3 neighborhood so 8 points; distance of 1 pixel
    lbp_hist, _ = np.histogram(lbp, 8)
    lbp_probabilities = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.sum(lbp_probabilities**2)
    lbp_entropy = -np.sum(np.multiply(lbp_probabilities, np.log2(lbp_probabilities)))
    lbpFeatures = [lbp_energy, lbp_entropy]

    return lbpFeatures

def getGaborFeatures(img):
    gaborFilter_real, gaborFilter_imag = gabor(img, frequency=0.6)
    gaborFilter = (gaborFilter_real**2 +gaborFilter_imag**2)/2
    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(gaborFilter_real , cmap='gray')
    # ax[1].imshow(gaborFilter_imag , cmap='gray')
    # ax[2].imshow(gaborFilter, cmap='gray')
    gabor_hist, _ = np.histogram(gaborFilter, 8)
    gabor_probabilities = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_probabilities**2)
    gabor_entropy = -np.sum(np.multiply(gabor_probabilities, np.log2(gabor_probabilities)))
    gaborFeatures =[gabor_energy, gabor_entropy]

    return gaborFeatures

def getRLFeatures(img):
    img = img.astype(np.uint8)  ## convert from 12 bit to 8 bit
    RLS = glrlm.getGLRLMStatistics(img)
    maxRLSFeatures = np.amax(RLS, axis=0)
    meanRLSFeatures = np.mean(RLS, axis=0)
    RLSFeatures = list(np.concatenate((maxRLSFeatures,meanRLSFeatures)))

    return RLSFeatures

def getTextureFeatures(img):
    GLCM_features  = getGLCMFeatures(img)
    RL_features    = getRLFeatures(img)
    #gabor_features = getGaborFeatures(img)
    #LBP_features   = getLBPFeatures(img)

    #features = [GLCM_features, RL_features, gabor_features, LBP_features]
    features = [GLCM_features, RL_features]
    features = [item for sublist in features for item in sublist]
    return features


# img = np.load("testingImage.npy")
# RLFeatures = getRLFeatures(img)
# GLCMFeatures = getGLCMFeatures(img)

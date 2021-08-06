import openImage
import numpy as np
import matplotlib.pyplot as plt

imageFilePath = "C:\\Users\jones\Desktop\MammData\FFDM_CCOnly\ARP0001_0701_LCC.img"
#
imgRaw = openImage.load_image(imageFilePath)

def segment(filePath, image):

    #def segmentCC(image):
    nrows,ncolumns = image.shape
    padding = 20
    LorR = filePath[-7]

    if LorR == 'R':
        #print("this is a right image.")
        ##get the right most column
        columnZero = image[:,ncolumns-2]
        nonZeroEntry = columnZero.nonzero()
        topBound = nonZeroEntry[0][0]
        bottomBoundIndex = len(nonZeroEntry[0])
        bottomBound = nonZeroEntry[0][bottomBoundIndex-1]

        ## get the side edge
        for i in range(ncolumns-1):
            x = not np.any(image[:,ncolumns-2-i])
            if x is True:
                #print("at the edge at column," , i)
                edgeBound = i
                break

        ##get new image dimensions
        top = topBound - padding
        bottom = bottomBound + padding
        edge = edgeBound - padding
        img = image[top:bottom + 1, edge:ncolumns-1]

    else:
        #print("this is a left image")
        ##get the left most column
        columnZero = image[:,0]
        nonZeroEntry = columnZero.nonzero()
        topBound = nonZeroEntry[0][0]
        bottomBoundIndex = len(nonZeroEntry[0])
        bottomBound = nonZeroEntry[0][bottomBoundIndex-1]

        ##get new image dimensions
        top = topBound - padding
        bottom = bottomBound + padding
        img = image[top:bottom + 1, :]

        ##get the side edge
        for i in range(img.shape[1]-1):
            x = not np.any(img[:,i])
            if x is True:
                #print("at the edge at column," , i)
                edgeBound = i
                break

        ##get new image dimensions
        # top = topBound - padding
        # bottom = bottomBound + padding
        edge = edgeBound + padding
        img = img[:, 0:edge + 1]

    return img, LorR
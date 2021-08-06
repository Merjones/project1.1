import openImage
import segmentImage
import math
import numpy as np
import matplotlib.pyplot as plt

## "Texture information in run-length matrices - Xiaoou Tang
## https://www.lifexsoft.org/index.php/resources/19-texture/radiomic-features/68-grey-level-run-length-matrix-glrlm
#  Short Run Emphasis (SRE)
#   Long Run Emphasis (LRE)
#   Gray-Level Nonuniformity (GLN)
#   Run Length Nonuniformity (RLN)
#   Run Percentage (RP)
#   Low Gray-Level Run Emphasis (LGRE)
#   High Gray-Level Run Emphasis (HGRE)
#   Short Run Low Gray-Level Emphasis (SRLGE)
#   Short Run High Gray-Level Emphasis (SRHGE)
#   Long Run Low Gray-Level Emphasis (LRLGE)
#   Long Run High Gray-Level Emphasis (LRHGE)

def find_runs(x): ## from alimanfoo on github
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def getAngledRows(image,angle, rowChange, columnChange):
    nrows, ncolumns = image.shape
    # for rowIndex in reversed(range(nrows)):
    overall = []
    for columnIndex in reversed(range(ncolumns)):
        rowIndex = nrows - 1
        #print("current" , rowIndex, columnIndex, image[rowIndex][columnIndex])
        ag = []
        while (rowIndex >= 0) and (0 <= columnIndex<ncolumns):
            ag.append(image[rowIndex][columnIndex])
            rowIndex = rowIndex + rowChange
            columnIndex = columnIndex + columnChange
        overall.append(ag)

    if angle == '45':
        for rowIndex in reversed(range(nrows - 1)):
            columnIndex = 0
            ag = []
            while (rowIndex >= 0) and (0 <= columnIndex < ncolumns):
                ag.append(image[rowIndex][columnIndex])
                rowIndex = rowIndex + rowChange
                columnIndex = columnIndex + columnChange
            overall.append(ag)
    elif angle == '135':
        for rowIndex in reversed(range(nrows-1)):
            columnIndex = image.shape[1] - 1
            ag = []
            while (rowIndex >= 0) and (0 <= columnIndex < ncolumns):
                ag.append(image[rowIndex][columnIndex])
                rowIndex = rowIndex + rowChange
                columnIndex = columnIndex + columnChange
            overall.append(ag)

    return overall

## default angles are 0,45,90,135
def getGLRLM(image, Ni, Nr, greyValues):

    glrlm = []

    angles = {"45": [-1, 1],
              "90": [-1, 0],
              "135": [-1, -1]}
    ## for the 0 angle
    zeroAngle = np.zeros((Ni, Nr))
    for row in image:
        g = find_runs(row)
        for i in range(len(g[0])):
            greyLevelValue = g[0][i]
            runLength = g[2][i]  # runLength
            columnIndex = runLength - 1
            rowIndex = np.where(greyValues == greyLevelValue)[0][0]
            zeroAngle[rowIndex][columnIndex] += 1

    glrlm.append(zeroAngle)

    for key,item in angles.items():
        id = np.zeros((Ni, Nr))
        tempRows = getAngledRows(image, key, item[0],item[1])
        for row in tempRows:
            g = find_runs(row)
            for i in range(len(g[0])):
                greyLevelValue = g[0][i]
                runLength = g[2][i]  # runLength
                columnIndex = runLength - 1
                rowIndex = np.where(greyValues == greyLevelValue)[0][0]
                id[rowIndex][columnIndex] += 1
        glrlm.append(id)

    # print(glrlm)
    return glrlm

def getRunLengthStatistics(glrlm,N_p):
    cVector = np.arange(1, glrlm.shape[1] + 1)
    rVector = np.arange(1, glrlm.shape[0] + 1)
    [cMatrix, rMatrix] = np.meshgrid(cVector, rVector)
    cMatrix = cMatrix.astype(np.int64)
    rMatrix = rMatrix.astype(np.int64)

    P_g = np.sum(glrlm, axis=1)  ##gray level run-number vector
    P_r = np.sum(glrlm, axis=0)  ##run-length run-number vector
    N_r = np.sum(glrlm)  ##total number of runs

    SRE = sum(P_r / (cVector ** 2)) / N_r
    LRE = sum(P_r * cVector ** 2) / N_r
    GLN = sum(P_g ** 2) / N_r
    RLN = sum(P_r ** 2) / N_r
    RP  = N_r / N_p
    LGRE = sum(P_g / (rVector ** 2)) / N_r
    HGRE = sum(P_g * (rVector ** 2)) / N_r
    SRLGE = np.sum(glrlm / ((rMatrix*cMatrix)**2)) / N_r
    SRHGE = np.sum(glrlm * ((rMatrix**2)) / (cMatrix**2)) / N_r
    LRLGE = np.sum(glrlm * ((cMatrix ** 2) / (rMatrix ** 2))) / N_r
    LRHGE = np.sum(glrlm * (rMatrix ** 2) * (cMatrix ** 2)) / N_r

    RLS = [SRE, LRE, GLN, RLN, RP, LGRE, HGRE, SRLGE, SRHGE, LRLGE, LRHGE]

    return RLS

def getGLRLMStatistics(img):
    greyValues = np.unique(img)  ##gray levels in the image
    N_i = len(greyValues)  ##number of discrete gray levels in the image
    N_rMax = np.max(img.shape)  ##max possible run length
    N_p = img.size  ##number of pixels in the image

    allGLRLM = getGLRLM(img, N_i, N_rMax, greyValues)
    statistics = np.zeros([11])
    for GLRLM in allGLRLM:
        stats = getRunLengthStatistics(GLRLM, N_p)
        statistics = np.vstack([statistics, stats])

    statistics = np.delete(statistics, 0, 0)
    return statistics

# #x = np.array([[0,255,113,113,42], [255,42,113,113,0],[128,113,255,0,42],[113,255,0,128,42],[42,113,128,255,255]])

# x = np.array([[1,1,1,2,2],[3,4,2,2,3],[4,4,4,4,4],[5,5,3,3,3],[1,1,3,4,5]])

# imageFilePath = "C:\\Users\jones\Desktop\MammData\Test-Images-GE-format\M2000210_1612_LCC.img"
# imgRaw = openImage.load_image(imageFilePath)
# img1 = imgRaw.astype(np.uint8)
# img, lorr = segmentImage.segmentImage(imageFilePath, img1)
# #
# # #x = np.array([[0,1,1,1,3,4,5,5,1], [2,2,2,2,2,3,1,1,1],[0,0,0,0,2,5,5,5,5]])
# #
# y = getGLRLMStatistics(img)
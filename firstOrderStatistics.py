import numpy as np

## decsribes the distribution of voxel intensities within the image
def getMean(image):
    return np.mean(image)

def getMax(image):
    return np.max(image)

def getStdev(image):
    return np.std(image)

def getEnergy(image):
    return np.sum(image**2)

def getEntropy(image):
    hist, _ = np.histogram(image)
    prob    = np.divide(hist, np.sum(hist))
    entropy =  -np.sum(np.multiply(prob, np.log2(prob)))
    return entropy

def getSkewness(image):
    mean = np.mean(image)
    num = np.mean(((image - mean) ** 3))
    den = (np.sqrt(np.mean(((image - mean) ** 2)))) ** 3
    return (num/den)

def getKurtosis(image):
    mean = np.mean(image)
    num = np.mean(((image - mean) ** 4))
    den = (np.sqrt((np.mean((image - mean) ** 2)))) ** 2
    return (num/den)

def computeFirstOrder(image):
    mean = getMean(image)
    max = getMax(image)
    stdev = getStdev(image)
    energy = getEnergy(image)
    entropy = getEntropy(image)
    skewness = getSkewness(image)
    kurtosis = getKurtosis(image)

    if (mean == 0):
        print("this image is trash")

    return mean, max, stdev, energy, entropy, skewness, kurtosis
    #return mean, max, stdev, energy







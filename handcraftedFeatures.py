import firstOrderStatistics
import textureFeatures

def getFeatures(img):
    firstOrder_features = firstOrderStatistics.computeFirstOrder(img)
    texture_features = textureFeatures.getTextureFeatures(img)

    handcrafted_features = [firstOrder_features,texture_features]
    handcrafted_features = [item for sublist in handcrafted_features for item in sublist]
    return handcrafted_features


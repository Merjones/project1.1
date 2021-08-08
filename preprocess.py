import numpy as np
import pandas as pd
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
# automatedFeat   = pd.read_pickle("allAutomatedFeatures.pkl")
#
def standardize(featureDF):
    # independentVarLength = (len(featureDF.columns) - 1)
    # x = featureDF.iloc[:, :independentVarLength]
    # y = featureDF.iloc[:, independentVarLength]

    sc = StandardScaler()
    sc.fit(featureDF)
    features = sc.transform(featureDF)

    return  features

    # independentVarLength = (len(automatedFeat.columns) - 1)
    # x_automated_ = automatedFeat.iloc[:, :independentVarLength]
    # y_automated = automatedFeat.iloc[:, independentVarLength]
    #
    # sc_automated = StandardScaler()
    # sc_automated.fit(x_automated_)
    # x_automated = sc_automated.transform(x_automated_)
    #
    # sc_handcrafted = StandardScaler()
    # sc_handcrafted.fit(x_handcrafted_)
    # x_handcrafted = sc_handcrafted.transform(x_handcrafted_)
    #
    # # pca - keep 90% of variance
    # # pca = PCA(0.90)
    # #
    # # print("automated pre PCA:", x_automated.shape)
    # # principal_components_automated = pca.fit_transform(x_automated)
    # # x_automated = pd.DataFrame(data = principal_components_automated)
    # # print("automated post PCA:", x_automated.shape)
    #
    # x_merged = pd.concat([x_handcrafted_, x_automated_], axis=1)
    #
    # sc_merged = StandardScaler()
    # sc_merged.fit(x_merged)
    # x_merged = sc_merged.transform(x_merged)
    #
    # # print("handcrafted pre PCA:", x_handcrafted.shape)
    # # principal_components_handcrafted = pca.fit_transform(x_handcrafted)
    # # principal_df_handcrafted = pd.DataFrame(data = principal_components_handcrafted)
    # # print("handcrafted post PCA:", principal_df_handcrafted.shape)
    #
    # return x_handcrafted, x_automated, x_merged, y_automated
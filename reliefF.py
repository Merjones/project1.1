from ReliefF import ReliefF
import numpy as np
import pandas as pd

handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
# automatedFeat   = pd.read_pickle("autoFeat_reduced_preRelief.pkl")
# mergedFeat      = pd.read_pickle("mergedFeat_reduced_preRelief.pkl")
automatedFeat   = pd.read_pickle("autoFeatVarianceThresh0.pkl")
automatedFeat_v2     = pd.read_pickle("autoFeatVarThresh0Drop20unique.pkl")
massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0', 'Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map(
    {1: 1, 2: 1, 3: -1, 5: -1})  # mapping all malignancies to a 1 and all benign to a -1
massType = massCenterMarks['Mass Type']

fs = ReliefF(n_neighbors=1, n_features_to_keep=20)
x_T = fs.fit_transform(automatedFeat,massType)
print(x_T)
print("(No. of tuples, No. of Columns before ReliefF) : "+str(automatedFeat.shape)+
      "\n(No. of tuples , No. of Columns after ReliefF) : "+str(x_T.shape))
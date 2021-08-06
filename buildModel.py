import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import preprocess

massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0','Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])

imageNames = massCenterMarks['image_name']
massType = massCenterMarks['Mass Type']
handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
automatedFeat   = pd.read_pickle("allAutomatedFeatures.pkl")

x_handcrafted ,x_automated, x_merged, y = preprocess.reduceFeatures(handcraftedFeat,automatedFeat)
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map({1:1, 2:1, 3:3, 5:3})

handCrafted = pd.concat([pd.DataFrame(x_handcrafted), massType], axis=1)
automated = pd.concat([pd.DataFrame(x_automated), massType], axis=1)
merged = pd.concat([pd.DataFrame(x_merged) , massType], axis=1)

clf = svm.SVC(kernel='linear', C=1)

handcraftedScores = cross_val_score(clf, x_handcrafted,y,cv=10)
automatedScores = cross_val_score(clf, x_automated,y, cv=10)
mergedScores = cross_val_score(clf, x_merged, y, cv=10)
print("Handcrafted average:", handcraftedScores.mean())
print("Automated average:", automatedScores.mean())
print("Merged average:", mergedScores.mean())

# handCrafted['Mass Type'] = handCrafted['Mass Type'].map({1:1, 2:1, 3:0, 5:0})
# automated['Mass Type'] = automated['Mass Type'].map({1:1, 2:1, 3:0, 5:0})
# merged['Mass Type'] = merged['Mass Type'].map({1:1, 2:1, 3:0, 5:0})



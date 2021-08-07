import numpy as np
import pandas as pd
import os
import getFeatures
import preprocess
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve, roc_curve, auc, roc_auc_score



def getAverage(scoresList):
    score = 0
    for s in scoresList:
        score += s
    return (score/len(scoresList))

##STEP 1
# load the data
imageFolderPath = "ccOnly"
massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0','Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map({1:1, 2:1, 3:-1, 5:-1}) #mapping all malignancies to a 1 and all benign to a -1

###STEP 2
#get the features
#handcraftedFeat, automatedFeat =  getFeatures.features(massCenterMarks, imageFolderPath)
# handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
# automatedFeat   = pd.read_pickle("allAutomatedFeatures.pkl")
# mergedFeat = pd.concat([handcraftedFeat, automatedFeat], axis=1)

automatedFeat = pd.read_pickle("autoFeat_reduced_preRelief.pkl")
mergedFeat    = pd.read_pickle("mergedFeat_reduced_preRelief.pkl")

###STEP 3
# preprocess
x_handcrafted = preprocess.reduceFeatures(handcraftedFeat)
x_automated   = preprocess.reduceFeatures(automatedFeat)
x_merged      = preprocess.reduceFeatures(mergedFeat)
massType = massCenterMarks['Mass Type']


DATA = [ x_handcrafted, x_automated, x_merged]
# = [x_automated, x_merged]
cv = KFold(n_splits=10,random_state=1, shuffle=True)

results = pd.DataFrame(columns=(list(range(0, 11))))
results.columns = [*results.columns[:-1], 'Average']

for d in DATA:
    print(d)
    model = svm.SVC(kernel='linear', C=1, probability=True)
    accuracyScores = []
    AUCscores = []
    ROCAUCScores = []

    for train_index, test_index in cv.split(d):
        x_train = d[train_index]
        x_test  = d[test_index]
        y_train = massType.loc[train_index]
        y_test  = massType.loc[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracyScores.append(accuracy)
        fpr , tpr, _ = roc_curve(y_test,predictions)
        aucH = auc(fpr,tpr)
        AUCscores.append(aucH)
        print("aucH using metrics.auc is:", aucH)
        rocAUC = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
        ROCAUCScores.append(rocAUC)
        print("roc auc is:", rocAUC)

    avgAccuracy = getAverage(accuracyScores)
    avgAUC      = getAverage(AUCscores)
    avgROCAUC   = getAverage(ROCAUCScores)

    accuracyScores.append(avgAccuracy)
    AUCscores.append(avgAUC)
    ROCAUCScores.append(avgROCAUC)
    results.loc[len(results)] = accuracyScores
    results.loc[len(results)] = AUCscores
    results.loc[len(results)] = ROCAUCScores









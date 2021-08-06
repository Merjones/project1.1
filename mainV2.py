import numpy as np
import pandas as pd
import os
import getFeatures
import preprocess
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve

##STEP 1
# load the data
imageFolderPath = "ccOnly"
massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0','Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map({1:1, 2:1, 3:3, 5:3}) #mapping all malignancies to a 1 and all benign to a 3

###STEP 2
#get the features
#handcraftedFeat, automatedFeat =  getFeatures.features(massCenterMarks, imageFolderPath)
handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
automatedFeat   = pd.read_pickle("allAutomatedFeatures.pkl")
mergedFeat = pd.concat([handcraftedFeat, automatedFeat], axis=1)

###STEP 3
# preprocess
x_handcrafted = preprocess.reduceFeatures(handcraftedFeat)
x_automated   = preprocess.reduceFeatures(automatedFeat)
x_merged      = preprocess.reduceFeatures(mergedFeat)

massType = massCenterMarks['Mass Type']

handCrafted = pd.concat([pd.DataFrame(x_handcrafted), massType], axis=1)
automated = pd.concat([pd.DataFrame(x_automated), massType], axis=1)
merged = pd.concat([pd.DataFrame(x_merged) , massType], axis=1)

clf = svm.SVC(kernel='linear', C=1)
#
handcraftedScores = cross_val_score(clf, x_handcrafted,massType,cv=10)
automatedScores = cross_val_score(clf, x_automated,massType, cv=10)
mergedScores = cross_val_score(clf, x_merged, massType, cv=10)
print("Handcrafted average:", handcraftedScores.mean())
print("Automated average:", automatedScores.mean())
print("Merged average:", mergedScores.mean())


cv = KFold(n_splits=5,random_state=1, shuffle=True)

modelHandcrafted = svm.SVC(kernel='linear', C=1)
modelAutomated = svm.SVC(kernel='linear', C=1)
modelMerged = svm.SVC(kernel='linear', C=1)

handcraftedScores = []
handcraftedROCAUC = []
automatedScores = []
automatedROCAUC = []
mergedScores = []
mergedROCAUC = []
count = 0

tprs= []
aucs=[]
mean_fpr = np.linspace(0,1,100)
fig,ax = plt.subplots(3)

for i, (train,test) in enumerate(cv.split(handCrafted)):
    #x_trainNames, x_testNames = imageNames[train], imageNames[test]
    trainingSet = handCrafted.loc[train]
    testingSet = handCrafted.loc[test]
    independentVarLength = (len(handCrafted.columns) - 1)
    x_train = trainingSet.iloc[:, :independentVarLength]
    y_train = trainingSet.iloc[:, independentVarLength]
    x_test = testingSet.iloc[:, :independentVarLength]
    y_test = testingSet.iloc[:, independentVarLength]

    modelHandcrafted.fit(x_train, y_train)
    viz = plot_roc_curve(modelHandcrafted, x_test, y_test, name='ROC fold {}'.format(i),
                         alpha = 0.3, lw=1, ax=ax[0])
    interp_tpr = np.interp(mean_fpr,viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    handcraftedROCAUC.append(viz.roc_auc)

    predictions = modelHandcrafted.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    handcraftedScores.append(accuracy)
    count+=1
    print("iteration:", count, "in handcrafted")

ax[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)


tprs= []
aucs=[]
count = 0
for i, (train,test) in enumerate(cv.split(automated)):
    #x_trainNames, x_testNames = imageNames[train], imageNames[test]
    trainingSet = automated.loc[train]
    testingSet = automated.loc[test]
    independentVarLength = (len(automated.columns) - 1)
    x_train = trainingSet.iloc[:, :independentVarLength]
    y_train = trainingSet.iloc[:, independentVarLength]
    x_test = testingSet.iloc[:, :independentVarLength]
    y_test = testingSet.iloc[:, independentVarLength]

    modelAutomated.fit(x_train, y_train)
    viz = plot_roc_curve(modelAutomated, x_test, y_test, name='ROC fold {}'.format(i),
                         alpha = 0.3, lw=1, ax=ax[1])
    interp_tpr = np.interp(mean_fpr,viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    automatedROCAUC.append(viz.roc_auc)

    predictions = modelAutomated.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    automatedScores.append(accuracy)
    count+=1
    print("iteration:", count, "in automated")

ax[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

tprs= []
aucs=[]
count= 0
for i, (train,test) in enumerate(cv.split(merged)):
    #x_trainNames, x_testNames = imageNames[train], imageNames[test]
    trainingSet = merged.loc[train]
    testingSet = merged.loc[test]
    independentVarLength = (len(merged.columns) - 1)
    x_train = trainingSet.iloc[:, :independentVarLength]
    y_train = trainingSet.iloc[:, independentVarLength]
    x_test = testingSet.iloc[:, :independentVarLength]
    y_test = testingSet.iloc[:, independentVarLength]

    modelMerged.fit(x_train, y_train)
    viz = plot_roc_curve(modelMerged, x_test, y_test, name='ROC fold {}'.format(i),
                         alpha = 0.3, lw=1, ax=ax[2])
    interp_tpr = np.interp(mean_fpr,viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    mergedROCAUC.append(viz.roc_auc)
    predictions = modelMerged.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    mergedScores.append(accuracy)
    count+=1
    print("iteration:", count, "in merged")

ax[2].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
               label='Chance', alpha=.8)


plt.show()
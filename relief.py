import numpy as np
import pandas as pd
import random
from sklearn import svm
import preprocess
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve, roc_auc_score, auc, roc_curve
from sklearn.feature_selection import VarianceThreshold

def getAverage(scoresList):
    score = 0
    for s in scoresList:
        score += s
    return (score/len(scoresList))

def euclideanDistance(vector1, vector2):
    d = ((vector1 - vector2)**2).sum()
    return np.sqrt(d)

def manhattanDistance(vector1, vector2):
    m = abs(vector1-vector2).sum()
    return m

def getNearest(featureSet,trainingCaseVector):
    distanceList = []
    for i, row in enumerate(featureSet):
        dist = manhattanDistance(trainingCaseVector, row)
        distanceList.append((i,dist))
    return distanceList

def diff(I1, I2, feature): ##I1 is the target instance and I2 is either nH or nM
    d = (I1 - I2)/(feature.max()-feature.min())
    return d

def reduce(featureDF,featureLabel,m,num): #m is number of training examples and n is number of desired features
    n = featureDF.shape[0]  # number of total training instances
    a = featureDF.shape[1]  # number of

    ##use minmaxscaler before calculating the distance
    scaler = MinMaxScaler()
    scaler.fit(featureDF)
    features = scaler.transform(featureDF)

    print("Using ", m, " training examples to get ", num, " features.")

    featureWeights = np.zeros((a))  # initalize all feature weights to 0
    for i in range(m):
        targetIndex = random.randint(0, n - 1)
        targetInstance = features[targetIndex]
        targetLabel = featureLabel[targetIndex]
        distList = getNearest(features, targetInstance)
        distList.sort(key=lambda x: x[1])  ##sorts the dist list by smallest to largest
        distList.pop(0)  ## pop off the first element which will be the target instance
        targets = [1, -1]

        for item in distList:
            if targets == []:
                break
            else:
                index = item[0]
                label = featureLabel[index]
                if label in targets:
                    if label == targetLabel:
                        nH_index = index
                    else:
                        nM_index = index
                    targets.remove(label)

        nH = features[nH_index]
        nM = features[nM_index]

        ##update weights
        for i, weight in enumerate(featureWeights):
            feature = featureDF[i]
            featureWeights[i] = weight - (diff(targetInstance[i], nH[i], feature) / m) + (
                    diff(targetInstance[i], nM[i], feature) / m)

    featureWeightSortedIndex = np.argsort(featureWeights)[::-1]

    selectedFeatures = pd.DataFrame()

    for i in range(num):
        desiredColumnIndex = featureWeightSortedIndex[i]
        selectedFeatures[i] = mergedFeat[desiredColumnIndex]

    return  selectedFeatures






massCenterMarks = pd.read_excel('onlyCC.xlsx')
massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0', 'Unnamed: 5', 'Unnamed: 6',
                                                'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                'Unnamed: 10'])
massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map(
    {1: 1, 2: 1, 3: -1, 5: -1})  # mapping all malignancies to a 1 and all benign to a -1
massType = massCenterMarks['Mass Type']
handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")
automatedFeat   = pd.read_pickle("autoFeat_reduced_preRelief.pkl")
mergedFeat      = pd.read_pickle("mergedFeat_reduced_preRelief.pkl")

handcraftedFeaturesReduced = reduce(handcraftedFeat,massType, 1000, 20 )
automatedFeaturesReduced   = reduce(automatedFeat, massType, 1000,1000)
mergedFeaturesReduced   = reduce(mergedFeat, massType, 1000,1000)

handcraftedFeaturesReduced.to_pickle("handcraftedFeat_postRelief.pkl")
automatedFeaturesReduced.to_pickle("automatedFeat_postRelief.pkl")
mergedFeaturesReduced.to_pickle("mergedFeat_postRelief.pkl")


x_handcrafted = preprocess.reduceFeatures(handcraftedFeaturesReduced)
x_automated   = preprocess.reduceFeatures(automatedFeaturesReduced)
x_merged      = preprocess.reduceFeatures(mergedFeaturesReduced)
massType = massCenterMarks['Mass Type']

DATA = [ x_handcrafted, x_automated, x_merged]
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

results.to_pickle("resultsPostRelief.pkl")
#
# def getResults():
#     massCenterMarks = pd.read_excel('onlyCC.xlsx')
#     massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0', 'Unnamed: 5', 'Unnamed: 6',
#                                                     'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
#                                                     'Unnamed: 10'])
#     massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map(
#         {1: 1, 2: 1, 3: -1, 5: -1})  # mapping all malignancies to a 1 and all benign to a -1
#     massType = massCenterMarks['Mass Type']
#
#     ###############
#     # automatedFeat_ = pd.read_pickle("allAutomatedFeatures.pkl")
#     #
#     # #normalize all features by dividing by the mean in order to compare their variances
#     # #normalizedAutoFeat = automatedFeat_ / automatedFeat_.mean()
#     # sc = StandardScaler()
#     # sc.fit(automatedFeat_)
#     # automatedFeat_= sc.transform(automatedFeat_)
#     #
#     # varianceSelector = VarianceThreshold(1)
#     # autoFeat = varianceSelector.fit_transform(automatedFeat_)
#     #
#     # ##turn back into pandas dataframe
#     # automatedFeat = pd.DataFrame(autoFeat)
#     #
#     # ##drop all columns that have less than 20 unique values
#     # count = 0
#     # for col in automatedFeat.columns:
#     #     if ((automatedFeat[col].nunique()) < 20):
#     #         automatedFeat.drop(col, inplace=True, axis=1)
#     #         count+=1
#     # print("Dropping this many colums:", count)
#
#     automatedFeat = pd.read_pickle('allAutomatedFeatures_REDUCED.pkl')
#     a_ = automatedFeat.shape[1]  # number of attributes
#
#     handcraftedFeat = pd.read_pickle('allHandcraftedFeatures.pkl')
#
#     ##rename columns of automated so can index later
#     colNames = range(0, a_)
#     automatedFeat.columns = colNames
#
#     mergedFeat = pd.concat([handcraftedFeat, automatedFeat], axis=1)
#
#     n = mergedFeat.shape[0]  # number of training instances
#     a = mergedFeat.shape[1]  # number of attributes
#
#     ##rename columns of automated so can index later
#     colNames = range(0, a)
#     mergedFeat.columns = colNames
#
#     m = 10  # number of random training examples out of n to use to update weights
#     m = [10,100,1000]
#     numMergedFeatures = [100,500,1000,2000]
#     results = []
#     for m_ in m:
#         for num in numMergedFeatures:
#             print("Using " , m_, " training examples and ", num, " features.")
#
#             featureWeights = np.zeros((a))  # initalize all feature weights to 0
#             for i in range(m_):
#                 targetIndex = random.randint(0, n-1)
#                 targetInstance = mergedFeat.iloc[targetIndex]
#                 targetLabel = massType[targetIndex]
#                 distList = getNearest(mergedFeat,targetIndex)
#                 distList.sort(key=lambda x: x[1])  ##sorts the dist list by smallest to largest
#                 distList.pop(0)  ## pop off the first element which will be the target instance
#                 targets = [1, 3]
#                 for item in distList:
#                     index = item[0]
#                     label = massType[index]
#                     if label in targets:
#                         if label == 1:
#                             near1 = index
#                         elif label == 3:
#                             near3 = index
#                         targets.remove(label)
#
#                 if targetLabel == massType[near1]:
#                     nH = mergedFeat.iloc[near1]
#                     nM = mergedFeat.iloc[near3]
#                 elif targetLabel == massType[near3]:
#                     nH = mergedFeat.iloc[near3]
#                     nM = mergedFeat.iloc[near1]
#                 #
#                 # print("near 1:", near1)
#                 # print("near 3:", near3)
#
#                 ##update weights
#                 for i, weight in enumerate(featureWeights):
#                     feature = mergedFeat[i]
#                     featureWeights[i] = weight - (diff(targetInstance[i], nH[i], feature) / m_) + (
#                                 diff(targetInstance[i], nM[i], feature) / m_)
#
#             featureWeightSortedIndex = np.argsort(featureWeights)[::-1]
#
#             selectedMergedFeatures = pd.DataFrame()
#
#             for i in range(num):
#                 desiredColumnIndex = featureWeightSortedIndex[i]
#                 col = mergedFeat[desiredColumnIndex]
#                 selectedMergedFeatures[i] =
#
#     return selectedMergedFeatures
#
# r = getResults()
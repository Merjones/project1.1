import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve
from sklearn.feature_selection import VarianceThreshold

def euclideanDistance(vector1, vector2):
    d = ((vector1 - vector2)**2).sum()
    # for i in range(len(vector1)):
    #     d += (vector1[i] - vector2[i])**2
    return np.sqrt(d)

def getNearest(featuresDF, index):
    distanceList = []
    trainingCaseVector = featuresDF.iloc[index]
    for i, row in featuresDF.iterrows():
        dist = euclideanDistance(trainingCaseVector, row)
        distanceList.append((i,dist))
    return distanceList

def diff(I1, I2, feature): ##I1 is the target instance and I2 is either nH or nM
    d = (I1 - I2)/(feature.max()-feature.min())
    return d

def getResults():
    massCenterMarks = pd.read_excel('onlyCC.xlsx')
    massCenterMarks = massCenterMarks.drop(columns=['Unnamed: 0', 'Unnamed: 5', 'Unnamed: 6',
                                                    'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                                                    'Unnamed: 10'])
    massCenterMarks['Mass Type'] = massCenterMarks['Mass Type'].map(
        {1: 1, 2: 1, 3: 3, 5: 3})  # mapping all malignancies to a 1 and all benign to a 3
    massType = massCenterMarks['Mass Type']

    ###############
    # automatedFeat_ = pd.read_pickle("allAutomatedFeatures.pkl")
    #
    # #normalize all features by dividing by the mean in order to compare their variances
    # #normalizedAutoFeat = automatedFeat_ / automatedFeat_.mean()
    # sc = StandardScaler()
    # sc.fit(automatedFeat_)
    # automatedFeat_= sc.transform(automatedFeat_)
    #
    # varianceSelector = VarianceThreshold(1)
    # autoFeat = varianceSelector.fit_transform(automatedFeat_)
    #
    # ##turn back into pandas dataframe
    # automatedFeat = pd.DataFrame(autoFeat)
    #
    # ##drop all columns that have less than 20 unique values
    # count = 0
    # for col in automatedFeat.columns:
    #     if ((automatedFeat[col].nunique()) < 20):
    #         automatedFeat.drop(col, inplace=True, axis=1)
    #         count+=1
    # print("Dropping this many colums:", count)

    automatedFeat = pd.read_pickle('allAutomatedFeatures_REDUCED.pkl')
    n = automatedFeat.shape[0]  # number of training instances
    a = automatedFeat.shape[1]  # number of attributes

    ##rename columns so can index later
    colNames = range(0, a)
    automatedFeat.columns = colNames

    m = 10  # number of random training examples out of n to use to update weights
    m = [10,100,1000]
    numAutomatedFeatures = [100,500,1000,2000]
    results = []
    for m_ in m:
        for num in numAutomatedFeatures:
            print("Using " , m_, " training examples and ", num, " features.")

            featureWeights = np.zeros((a))  # initalize all feature weights to 0
            for i in range(m_):
                targetIndex = random.randint(0, n-1)
                targetInstance = automatedFeat.iloc[targetIndex]
                targetLabel = massType[targetIndex]
                distList = getNearest(automatedFeat,targetIndex)
                distList.sort(key=lambda x: x[1])  ##sorts the dist list by smallest to largest
                distList.pop(0)  ## pop off the first element which will be the target instance
                targets = [1, 3]
                for item in distList:
                    index = item[0]
                    label = massType[index]
                    if label in targets:
                        if label == 1:
                            near1 = index
                        elif label == 3:
                            near3 = index
                        targets.remove(label)

                if targetLabel == massType[near1]:
                    nH = automatedFeat.iloc[near1]
                    nM = automatedFeat.iloc[near3]
                elif targetLabel == massType[near3]:
                    nH = automatedFeat.iloc[near3]
                    nM = automatedFeat.iloc[near1]
                #
                # print("near 1:", near1)
                # print("near 3:", near3)

                ##update weights
                for i, weight in enumerate(featureWeights):
                    feature = automatedFeat[i]
                    featureWeights[i] = weight - (diff(targetInstance[i], nH[i], feature) / m_) + (
                                diff(targetInstance[i], nM[i], feature) / m_)

            featureWeightSortedIndex = np.argsort(featureWeights)[::-1]

            selectedAutoFeatures = pd.DataFrame()

            for i in range(num):
                desiredColumnIndex = featureWeightSortedIndex[i]
                col = automatedFeat[desiredColumnIndex]
                selectedAutoFeatures[i] = col

            sc = StandardScaler()
            sc.fit(selectedAutoFeatures)
            features = sc.transform(selectedAutoFeatures)

            clf = svm.SVC(kernel='linear', C=1)
            #
            automatedScores = cross_val_score(clf, features, massType, cv=10)
            print("automated average using:", m, "training instances is:", automatedScores.mean())

            model = svm.SVC(kernel='linear', C=1, probability=True)
            accuracyScores = []
            count = 0
            cv = KFold(n_splits=10, random_state=1, shuffle=True)
            for train_index, test_index in cv.split(features):
                x_train = features[train_index]
                x_test = features[test_index]
                y_train = massType.loc[train_index]
                y_test = massType.loc[test_index]

                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracyScores.append(accuracy)

            averageAccuracy = 0
            for score in accuracyScores:
                averageAccuracy += score
            averageAccuracy = averageAccuracy / len(accuracyScores)
            result = [m_, num, automatedScores.mean(),averageAccuracy]
            print(result)
            results.append(result)
    return results

r = getResults()
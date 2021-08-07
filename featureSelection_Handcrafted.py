import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve, roc_auc_score



# sc = StandardScaler()
# sc.fit(handcraftedFeat)
# handcraftedFeat = sc.transform(handcraftedFeat)

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
    handcraftedFeat = pd.read_pickle("allHandcraftedFeatures.pkl")

    n = handcraftedFeat.shape[0]  # number of training instances
    a = handcraftedFeat.shape[1]  # number of attributes
    m = 10  # number of random training examples out of n to use to update weights
    m = [10,100,1000]
    numHandcraftedFeatures = 41
    numHandcraftedFeatures = [10,20,41]
    results = []
    for m_ in m:
        for num in numHandcraftedFeatures:
            print("Using " , m_, " training examples and ", num, " features.")

            featureWeights = np.zeros((a))  # initalize all feature weights to 0
            for i in range(m_):
                targetIndex = random.randint(0, n-1)
                targetInstance = handcraftedFeat.iloc[targetIndex]
                targetLabel = massType[targetIndex]
                distList = getNearest(handcraftedFeat,targetIndex)
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
                    nH = handcraftedFeat.iloc[near1]
                    nM = handcraftedFeat.iloc[near3]
                elif targetLabel == massType[near3]:
                    nH = handcraftedFeat.iloc[near3]
                    nM = handcraftedFeat.iloc[near1]
                #
                # print("near 1:", near1)
                # print("near 3:", near3)

                ##update weights
                for i, weight in enumerate(featureWeights):
                    feature = handcraftedFeat[i]
                    featureWeights[i] = weight - (diff(targetInstance[i], nH[i], feature) / m_) + (
                                diff(targetInstance[i], nM[i], feature) / m_)

            # handcraftedFeatureWeights = np.load("featureWeightsHandcrafted_100.npy")
            featureWeightSortedIndex = np.argsort(featureWeights)[::-1]

            selectedHandFeatures = pd.DataFrame()

            for i in range(num):
                desiredColumnIndex = featureWeightSortedIndex[i]
                col = handcraftedFeat[desiredColumnIndex]
                selectedHandFeatures[i] = col

            sc = StandardScaler()
            sc.fit(selectedHandFeatures)
            features = sc.transform(selectedHandFeatures)

            clf = svm.SVC(kernel='linear', C=1)
            handcraftedScores = cross_val_score(clf, features, massType, cv=10)
            print("handcrafted average using:", m, "training instances is:", handcraftedScores.mean())

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
            #print("The average ROC AUC using:", m, "random training instances is:", average)
            result = [m_, num, handcraftedScores.mean(), averageAccuracy]
            results.append(result)

    return  results


r = getResults()
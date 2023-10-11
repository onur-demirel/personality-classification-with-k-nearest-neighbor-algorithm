import numpy as np
import pandas as pd
from random import randint

# reading the data
data = pd.read_csv("16p.csv", encoding="cp1252")
# dropping the response id column
data.drop("Response Id", axis=1, inplace=True)

# encoding the personality column to numbers from 0 to 15 (16 personalities)
def encodePersonality(data):
    personalities = {"ESTJ": 0, "ENTJ": 1, "ESFJ": 2, "ENFJ": 3, "ISTJ": 4, "ISFJ": 5, "INTJ": 6, "INFJ": 7, "ESTP": 8, "ESFP": 9, "ENTP": 10, "ENFP": 11, "ISTP": 12, "ISFP": 13, "INTP": 14, "INFP": 15}
    if data in personalities:
        return personalities[data]
    else:
        return np.nan
# encoding the personality column
data["Personality"] = data["Personality"].apply(encodePersonality)

# normalizing the data between 0 and 1 except the personality column
def normalization(data):
    if data.name != "Personality":
        data = (data - (-3)) / (6)
    return data
# calculating the normalized version of the data
data_normalized = data.apply(normalization)

# turning the data into numpy arrays, one being normalized and the other not
arr = data.to_numpy()
arr_normalized = data_normalized.to_numpy()

def euclideanDist(train, test):
    # basically, (x-y)^2 = x^2 + y^2 - 2xy
    # calculating the sum of the squares of each row in train
    trainSquares = np.sum(train[:,:-1]**2, axis=1)
    # calculating the sum of the squares of each row in test
    testSquares = np.sum(test[:,:-1]**2, axis=1)
    # calculating the dot product of each row in train and test to get X*Y term
    trainTest = np.dot(test[:,:-1], train[:,:-1].T)
    # print(trainSquares.shape, testSquares.shape, trainTest.shape)
    # calculating the euclidean distance
    distance = np.sqrt(trainSquares + testSquares[:, np.newaxis] - 2*trainTest)
    # print(distance.shape)
    # print(trainSquares.shape, testSquares.shape, distance.shape)
    # print(dists)
    return distance

# knn function to get the most common personality weighted and unweighted
def knn(train, test, k, weighted=False):
    # calculating the euclidean distance
    dists = euclideanDist(train, test)
    # sorting the distances
    sortedDists = np.argpartition(dists, kth=k, axis=1)
    # print(sortedDists.shape)
    # getting the k nearest neighbors
    nearestNeighbours = train[sortedDists[:,:k]]
    # print(nearestNeighbours.shape)
    # getting the personality of the k nearest neighbors
    nnPersonality = nearestNeighbours[:,:,-1]
    mostComPers = []
    for i in range(len(nnPersonality)):
        # getting the most common personality
        if weighted:
            # getting the weights
            weights = 1/dists[i][sortedDists[i][:k]]
            # getting the weighted personality
            weightedPers = np.bincount(nnPersonality[i], weights=weights)
            # getting the most common personality
            mostComPers.append(np.argmax(weightedPers))
        else:
            # if the distances are equal, then a random one is chosen
            if len(np.unique(dists[i][sortedDists[i][:k]])) == 1:
                mostComPers.append(np.random.choice(nnPersonality[i]))
            else:
                mostComPers.append(np.bincount(nnPersonality[i]).argmax())
    return mostComPers

# 5 fold cross validation by passing all of the test data with calculation of accuracy for the fold, precision and recall for each personality type
# and the average accuracy, precision and recall for each k value
# also prints the incorrect ones
# weighted and unweighted
def crossValidation(data):
    kList = [1, 3, 5, 7, 9]
    # shuffling the data
    np.random.shuffle(data)
    # splitting the data into 5 folds
    folds = np.array_split(data, 5)
    for w in [False, True]:
        print(f"Weighted\t{w}")
        for k in kList:
            incOnesDict = {f"Fold {i+1}": {j: [] for j in range(16)} for i in range(5)}
            print(f"K Value\t\t{k}")
            accuracyK = 0
            precisionK = np.zeros(16)
            recallK = np.zeros(16)
            testPoolK = np.zeros(16)
            actualPoolK = np.zeros(16)
            for i in range(5):
                accuracyFold = 0
                precisionFold = np.zeros(16)
                recallFold = np.zeros(16)
                # getting the test data
                test = folds[i]
                # getting the train data
                train = np.concatenate(folds[:i] + folds[i+1:])
                testPoolFold = np.zeros(16)
                actualPoolFold = np.zeros(16)
                # getting the predicted personality
                predictedPersonalities = knn(train, test, k, weighted=w)
                for p in enumerate(predictedPersonalities):
                    actualPersonality = test[p[0]][-1]
                    # calculating the accuracy, precision and recall
                    if p[1] == actualPersonality:
                        # calculating the accuracy, precision and recall
                        accuracyFold += 1
                        precisionFold[int(p[1])] += 1
                        testPoolFold[p[1]] += 1
                        recallFold[actualPersonality] += 1
                        actualPoolFold[actualPersonality] += 1
                    else:
                        # calculating the precision, recall and incorrect ones
                        incOnesDict[f"Fold {i+1}"][actualPersonality].append((p[0], p[1]))
                        testPoolFold[p[1]] += 1
                        actualPoolFold[actualPersonality] += 1
                print(f"")
                print(f"Fold {i+1}\t\tAccuracy: {accuracyFold/len(test)}")
                # macro average of precision and recall
                print(f"\t\tPrecision: {np.sum(precisionFold/testPoolFold)/16}")
                print(f"\t\tRecall: {np.sum(recallFold/actualPoolFold)/16}")                
                accuracyK += accuracyFold
                precisionK += precisionFold
                recallK += recallFold
                testPoolK += testPoolFold
                actualPoolK += actualPoolFold
            print(f"K Value {k}\tAccuracy: {accuracyK/len(data)}")
            # macro average of precision and recall
            print(f"\t\tPrecision: {np.sum(precisionK/testPoolK)/16}")
            print(f"\t\tRecall: {np.sum(recallK/actualPoolK)/16}")
            randMistake = incOnesDict["Fold {}".format(randint(1, 5))][randint(0, 15)]
            if len(randMistake) > 0:
                randInd = randint(0, len(randMistake)-1)
                print(f"Random Mistake: at index {randMistake[randInd][0]} the predicted personality is {randMistake[randInd][1]} and the actual personality is {data[randInd][-1]}")
            else:
                print(f"Random Mistake: None")

# cross validation for the non-normalized data
print("Not Normalized Data")                
crossValidation(arr)
# cross validation for the normalized data
print("Normalized Data")
crossValidation(arr_normalized)
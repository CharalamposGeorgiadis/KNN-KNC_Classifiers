from euclideanDistance import euclideanDistance
import numpy as np
from collections import Counter
from tqdm import tqdm


# Class that implements the k-nearest neighbors algorithm for any k
class KNN:

    # Constructor
    # param k: Integer containing the number of neighbors
    # param x: Training samples
    # param y: labels of the training samples
    def __init__(self, k, x, y):
        self.x = x
        self.y = y
        self.k = k

    # Predict method for predicting the labels of the test set
    # param x_test: Test samples
    def predict(self, x_test):
        # List that will hold the predictions for each test sample
        predictions = []
        for i in tqdm(range(len(x_test)), desc="Calculating Distances", leave=False):
            # Array that holds the distances between one test sample and each training sample
            distances = np.zeros(len(self.x))
            # Calculating the distance between one test sample and each training sample
            for j in range(len(self.x)):
                distances[j] = euclideanDistance(x_test[i], self.x[j])

            # Finding the k-nearest neighbors of one test sample by sorting the 'distances' array. np.argsort returns
            # the indices of the sorted array instead of the array itself
            k_nearest_neighbors = np.argsort(distances)[:self.k]

            # Finding the labels of the k-nearest neighbors
            k_nearest_labels = [self.y[k] for k in k_nearest_neighbors]

            # Voting for the label of the test sample by finding the most common label of its k-nearest neighbors
            predicted_label = Counter(k_nearest_labels).most_common(1)
            # Adding the predicted label to the list of predictions
            predictions.append(predicted_label[0][0])
        return np.array(predictions)

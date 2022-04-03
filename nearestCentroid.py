from euclideanDistance import euclideanDistance
import numpy as np


# Class that implements the nearest class centroid algorithm
class NearestCentroid:

    # Constructor
    # param classes: Integer containing the number of classes for the current dataset
    # param x: Training samples
    # param y: Labels of the training samples
    def __init__(self, classes, x, y):
        self.x = x
        self.y = y
        self.classes = classes

        # Array of lists, where each list holds the training samples that belong to the same class
        self.samples_per_class = np.empty(classes, dtype=np.object)
        for i in range(classes):
            self.samples_per_class[i] = []

        # Array that holds the centroids for each class
        self.centroids = np.zeros((self.classes, self.x.shape[1]))

        # Adding each sample to its respective position on the array of lists
        for i in range(self.x.shape[0]):
            label = y[i]
            self.samples_per_class[label].append(self.x[i])

        # Calculating the centroid of each class
        for i in range(self.classes):
            for j in range(len(self.samples_per_class[i])):
                self.centroids[i] += self.samples_per_class[i][j]
            self.centroids[i] = self.centroids[i] / len(self.samples_per_class[i])

    # Predict method for predicting the labels of the test set
    # param x_test: Test samples
    def predict(self, x_test):
        # List that will hold the predictions for each test sample
        predictions = []
        for i in range(len(x_test)):
            # Array that holds the distances between one test sample and each centroid
            distances = np.zeros(self.classes)
            for j in range(self.classes):
                # Calculating the distance between one test sample and each centroid
                distances[j] = euclideanDistance(x_test[i], self.centroids[j])

            # Finding the nearest centroid of one test sample by sorting the 'distances' array. np.argsort returns the
            # indices of the sorted array instead of the array itself
            nearest_centroid = np.argsort(distances)[0]

            # Adding the predicted label to the list of predictions
            predictions.append(nearest_centroid)
        return np.array(predictions)

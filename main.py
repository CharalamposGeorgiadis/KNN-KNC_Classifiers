import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from knn import KNN
from nearestCentroid import NearestCentroid
from IPython.display import clear_output

# Loading the MNIST database of handwritten digits
# Training set
train_set = pd.read_csv('dataset/mnist_train.csv')
x_train = train_set.drop(columns=['label'])
y_train = train_set['label']
# Test set
test_set = pd.read_csv('dataset/mnist_test.csv')
x_test = test_set.drop(columns=['label'])
y_test = test_set['label']

# Normalizing the data
x_train = np.array(pd.DataFrame(MinMaxScaler().fit_transform(x_train)))
x_test = np.array(pd.DataFrame(MinMaxScaler().fit_transform(x_test)))

neighbors = int(input("Enter the number of neighbors for the KNN classifier:\n"))

# Evaluating the 1-Nearest Neighbor classifier
clear_output()
print(str(neighbors) + "-NEAREST NEIGHBOR EVALUATION...")
clear_output()
start_time = time.time()
# Initializing the classifier
classifier = KNN(neighbors, x_train, y_train)
# Predicting the labels of the test set
predictions = classifier.predict(x_test)
end_time = time.time()
knn_time = end_time - start_time
# Calculating the accuracy of the classifier on the test set
knn_accuracy = np.sum(predictions == y_test) / len(y_test)
clear_output()
print(f"Accuracy: {knn_accuracy * 100:.2f}% | Execution Time: {round(knn_time * 1000)} ms")
clear_output()

# Evaluating the Nearest Class Centroid classifier on the encoded dataset
print("NEAREST CLASS CENTROID EVALUATION...")
start_time = time.time()
# Initializing the classifier
clf = NearestCentroid(10, x_train, y_train)
# Predicting the labels of the test set
predictions = clf.predict(x_test)
end_time = time.time()
ncc_time = end_time - start_time
# Calculating the accuracy of the classifier on the test set
ncc_accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {ncc_accuracy * 100:.2f}% | Execution Time: {round(ncc_time * 1000)} ms")

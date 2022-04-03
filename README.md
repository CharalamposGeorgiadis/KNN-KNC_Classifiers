# KNN-KNC_Classifiers

Python project that implements and evaluates the K-Nearest Nighbor and Nearest Class Centroid classifiers from scratch on the MNIST dataset.

The dataset is loaded using the Pandas library and normalized using the MinMaxScaler function from scikit-learn.
The distances are calcualted using the Euclidean Distance, the calculation of which is sped up with the help of the Numba library.

The performances of the KNN (for k = 1 and k = 3) and NCC classifiers are listed below:

| Classifier | Accuracy | Execution TIme (Avg.) |
| ------------- | ------------- | -------------|
| 1-Nearest Neighbors | 96.91% | ~2200000 ms  |
| 3-Nearest Neighbors | 97.17% | ~2200000 ms  |
| Nearest Class Centroid | 82.03% | ~780 ms  |

# KNN-KNC_Classifiers

Python project that implements and evaluates the K-Nearest Nighbor and Nearest Class Centroid classifiers from scratch on the MNIST dataset.

The dataset is loaded using the Pandas library and normalized using the MinMaxScaler function from scikit-learn.
The distances are calcualted using the Euclidean Distance, the calculation of which is sped up with the help of the Numba library.

# About
I wrote this Python implementation of the k-nearest neighbors algorithm for multi-class classification as a way to gain experience with basic machine learning techniques. The code accepts two separate files: one containing the data points to be classified and one containing their labels, and determines the best way to reduce their dimensonality, as well as the best k value to use.

# The data
The practice data I used to build this algorithm are provided in the repository. The "breast" file contains the features--a matrix of expression levels of 19,138 different genes assessed by microarray from samples of 515 breast cancer patients. "pm50" contains the tumor subtype classifications for all of the samples. The program was written to accept files in this specific format as input, and would thus have to be slightly modified to accept different datasets.

# The algorithm
Due to the high dimensionality of the dataset, the program uses principal component analysis to reduce the number of features. A maximum number of principal components and a maximum value for k are chosen by the user, and the program determines the classificiation accuracy at each combination of PC and k, returning the best performing combination and a heatmap of each combination's performance.

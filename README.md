# About
I wrote this Python implementation of the k-nearest neighbors algorithm for multi-class classification as a way to gain experience with basic machine learning techniques. The code accepts two separate files: one containing the data points to be classified and one containing their labels, and determines the best way to reduce their dimensonality, as well as the best k value to use.

# The data
The practice data I used to build this algorithm are provided in the repository. The "breast" file contains the features--a matrix of expression levels of 19,138 different genes assessed by microarray from samples of 515 breast cancer patients. "pm50" contains the tumor subtype classifications for all of the samples. The program was written to accept files in this specific format as input, and would thus have to be slightly modified to accept different datasets.

# The algorithm
Due to the high dimensionality of the dataset, the program uses principal component analysis to reduce the number of features. A maximum number of principal components and a maximum value for k are chosen by the user, and the program determines the weighted F1 score for the model's predictions at each combination of PC and k, returning the best performing combination and a heatmap of each combination's performance.

It should be noted that the F1 score may not be the best metric for the breast cancer data provided, since there are only 8 examples of "Normal-like" tumors in the dataset, which causes the number of points labeled by the model as Normal-like to quickly drop to zero. This causes zero division issues with the calculation of the F1 score. I have only very recently begun studying methods for assessing the performance of multi-class classification models, so I used F1 score since my understanding is that it is the most commonly used approach. I plan to implement a better option as I learn more.

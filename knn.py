import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def nearest_neighbors(labels, distances, k=5):        
    """Finds the k-nearest neighbors of a point.
    
    Accepts an array of labels and a corresponding array of distances from the point of interest, both of the 
    same length. Finds the most common label of the k-nearest neighbors by ascending order of distance. Ties are
    broken by summing the distances of each label in the k-nearest and taking the label with the lowest sum. Note
    that it is not necessary to average the distances, as each label in a tie will always have the same
    frequency.
    Args:
        labels
            An array of point labels as strings.
        distances
            An array representing the distances between each of the labeled points and the point of interest.
        k
            The number of nearest neighbors to find.
    """
    
    def sort_by_second(data1, data2, reverse=False):
        """Sorts two NumPy arrays both by the same args such that the second array is in order.
        
        Args:
            data1 
                The first array.
            data2
                The second array.
            reverse 
                A bool which specifies whether data2 should be sorted into ascending (False) or descending (True)
                order.
        """
        
        if reverse == True:
            sorting_args = np.argsort(-data2)
        else:
            sorting_args = np.argsort(data2)
        
        sorted_data1, sorted_data2 = data1[sorting_args], data2[sorting_args]
        
        return sorted_data1, sorted_data2
    
    # Sorts all the labeled points in ascending order of distance and eliminates all but the k nearest. It is
    # important to note that the function assumes that the closest point will be the point itself at distance 0,
    # since this is not removed by the calculate_distances function. Here, it can be easily removed by taking the 
    # k+1 nearest points minus the first.
    
    sorted_labels, sorted_distances = sort_by_second(labels, distances, reverse=False)
    sorted_labels, sorted_distances = sorted_labels[1:k+1], sorted_distances[1:k+1]
                                                
    # Finds the unique labels and their frequencies, and sorts them in descending order of frequency
    
    unique_labels, unique_counts = np.unique(sorted_labels, return_counts=True)
    unique_labels, unique_counts = sort_by_second(unique_labels, unique_counts, reverse=True)
            
    # Ends the function and returns the first unique label if there is only one or if its count exceeds the second
    
    if len(unique_labels) == 1 or unique_counts[0] > unique_counts[1]:
        return unique_labels[0]
    
    # Makes a list of labels tied for most common, each with an number 0 which will hold their total distances
    
    top_labels = []
    for i in range(len(unique_counts)):
        if unique_counts[i] == unique_counts[0]:
            top_labels.append([unique_labels[i], 0])
            
    # Finds the total distances for each of the most common labels
    
    for i, label in enumerate(top_labels):
        for j, value in np.ndenumerate(sorted_labels):
            if label[0] == value:
                top_labels[i][1] += sorted_distances[j]
    
    # Sorts the most common labels in ascending order of total distance and returns the nearest
    
    top_labels = sorted(top_labels, key=lambda x: x[1])     
    return top_labels[0][0]

def calculate_distances(point, data):
    """Finds the Euclidean distances between a given coordinate point and a set of points.
    
    Accepts a point in n-dimensional space specified by an array of size n and a set of p data points specified 
    by an array of size (p,n)
    
    Arguments:
        point
            An array representing the Cartesian coordinates of a point.
        data
            An array representing a set of points in the same coordinate space as the given point.
    """
    
    distances = []
    
    for i in data:
        dist = np.subtract(point, i)
        dist = np.square(dist)
        dist = np.sum(dist)
        distances.append(dist ** 0.5)
    
    return np.array(distances)

print("Loading labels...")

labels = np.genfromtxt('/Users/exampleuser/location/pm50.csv', delimiter='\t', dtype='str', usecols=1, skip_header=1)

print("Labels loaded. Loading data. This may take a few moments...")

X = np.genfromtxt("/Users/exampleuser/location/breast.csv", dtype='str', delimiter='\t')

print()
print("Assessing model performance...")

# The original file for the data used to develop this program puts the different samples on the x-axis and the 
# genes on the y; transposing it brings it into agreement with the labels file, which has the samples on the y. 
# The sample and gene names are removed prior to inputting the data into the KNN finder, but are saved to their 
# own arrays in case they are needed. In this implementation of the program, they are not used further.

X = X.transpose()
sample_names = X[1:,0]
gene_names = X[0,1:]
X = np.array(X[1:,1:], dtype=np.float64)

# pc_limit is the maximum number of principal components (i.e. the maximum dimensionality) that will be tested,
# beginning at 1. k_limit is the maximum number of nearest neighbors (k) that will be tested. I have tried it out
# to a pc_limit of 70 and a k_limit of 100 with the example breast cancer files provided, which takes roughly 5 
# minutes to execute on my 2018 MacBook Pro.

pc_limit = 25
k_limit = 40

n_samples = len(X)

# The results array holds the weighted F1 scores for each combination of PCs and k

results = np.zeros((pc_limit, k_limit))
best_performance = 0
best_pc_k = (0,0)
best_preds = []
all_labels = np.unique(labels)

for n_pc in range(1, pc_limit + 1):
    pca = PCA(n_components=n_pc)
    pca_data = pca.fit_transform(X)
    
    # The distances between each point and the rest of the points are calculated only once for each set of
    # principal components, in order to save time relative to calculating them repeatedly as the program iterates
    # through k. Since calculate_distances returns the distances to every other point for a given point in the 
    # data as a 1D array, all_distances is a list of arrays of length n_samples, with each element representing 
    # all the distances between a given point and the other points in the data.
    
    all_distances = []
    for i in range(n_samples):
        all_distances.append(calculate_distances(pca_data[i], pca_data))

    for k in range(1, k_limit + 1):
        predictions = []
        
        for i in range(n_samples):            
            predictions.append(nearest_neighbors(labels, all_distances[i], k))
            
        predictions = np.array(predictions)
        performance = f1_score(labels, predictions, labels=all_labels, average='weighted', zero_division=1)
        print("PC:", n_pc, "\t", "k:", k, "\t", "Weighted F1:", round(performance, 3))
        results[n_pc - 1,k - 1] = performance
        
        # The best performing combination of PC and k is stored to be printed at the end of the run
        
        if performance > best_performance:
            best_performance = performance
            best_pc_k = (n_pc, k)
            best_preds = predictions
            
print()
print("===========")
print("  RESULTS")
print("===========")
print()
print(f"The best F1 score was {round(best_performance, 3)}, achieved at {best_pc_k[0]} principal components and a k of {best_pc_k[1]}")
print()
print("Statistical summary of best performer: ")
print()
all_labels = np.unique(labels)
report = classification_report(labels, best_preds, target_names=all_labels, zero_division=1)
print(report)

# The code below creates a heatmap of each model's weighted F1 score, with k on the x-axis and the # of principal 
# components on the y-axis. Subtracting 0.75 from the results before printing a heatmap and converting everything 
# negative to zero turns all the least promising areas of the space to a uniform red and accentuates the
# differences between the rest.

results = results - 0.75
results = np.maximum(results, 0)

plt.imshow(results, cmap='autumn', interpolation='nearest')
plt.title("Weighted F1 score")
plt.ylabel("# of principal components")
plt.xlabel("k")
plt.show()

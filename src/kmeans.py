from scipy.spatial.distance import cdist
import numpy as np


def base_python(X, k, iterations=1000):
    """k-means algorithm implemented fully with only base python.

    Parameters
    ----------
    X : list of lists of numerics, data in length of inner list
        dimensional space
    k : int, number of clusters
    iterations : number of times to update centroids

    Returns
    -------
    centroids : list of list of numerics, length = k, dimensions same as X
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it
    """
    centroids = X[:k]
    for _ in range(iterations):
        centroid_assignments = get_new_assignments(centroids, X)
        centroids = calculate_new_centroids(centroid_assignments)
    return centroids, centroid_assignments


def get_new_assignments(centroids, X):
    """Helper function to find closest centroid for each data point.

    Parameters
    ----------
    centroids : list of list of numerics, length = k, dimensions same as X
    X : list of lists of numerics, data in length of inner list
        dimensional space
    
    Returns
    -------
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it
    """
    centroid_assignments = [[] for _ in centroids]
    for data_point in X:
        closest_dist = 1e100
        closest_centroid = None
        for centroid_idx, centroid_location in enumerate(centroids):
            current_dist = list_euclidean_dist(centroid_location, data_point)
            if current_dist < closest_dist:
                closest_dist = current_dist
                closest_centroid = centroid_idx
        centroid_assignments[closest_centroid].append(data_point)
    return centroid_assignments


def list_euclidean_dist(a, b):
    """Helper function to find the distance between two points represented as lists.

    Parameters
    ----------
    a : list of numerics
    b : list of numerics

    Returns
    -------
    float
    """
    return sum((da - db) ** 2 for da, db in zip(a, b)) ** 0.5


def calculate_new_centroids(centroid_assignments):
    """Helper function to find new centroid locations based on new centroid
    assignments.

    Parameters
    ----------
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it

    Returns
    -------
    new_centroids : list of list of numerics, length = k, dimensions same as X
    """
    new_centroids = []
    for centroid_assignment in centroid_assignments:
        centroid = []
        for dim in zip(*centroid_assignment):
            centroid.append(sum(dim) / len(dim))
        new_centroids.append(centroid) 
    return new_centroids


def numpy(X, k, iterations=1000):
    """k-means algorithm implemented with numpy.

    Parameters
    ----------
    X : ndarray, 2 dimensions
    k : int, number of clusters
    iterations : number of times to update centroids

    Returns
    -------
    centroids : ndarray, 2 dimesions
    centroid_assignments: list of ndarrays, points assigned to each centroid
                          by index
    """
    centroids = X[:k]
    for _ in range(iterations):
        closest_centroid_idxs = cdist(X, centroids).argmin(axis=1)
        for idx in range(k):
            centroids[idx] = np.mean(X[closest_centroid_idxs == idx], axis=0)

    centroid_assignments = make_centroid_assignments(X, closest_centroid_idxs)
    return centroids, centroid_assignments


def make_centroid_assignments(X, closest_idxs):
    """Helper function to divide up data by centroid assignment.

    Parameters
    ----------
    X : ndarray, 2 dimensions
    closest_idxs: ndarray, 1 dimension. Index of closest centroid for each
                  data point

    Returns
    -------
    centroid_assignments: list of ndarrays, points assigned to each centroid
    """
    return [X[closest_idxs == idx] for idx in np.unique(closest_idxs)]

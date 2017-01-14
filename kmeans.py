def base_python(X, k, iterations=1000):
    """k-means algorithm implemented fully with only base python.

    Parameters
    ----------
    X : list of lists of numerics, data in length of inner list
        dimensional space
    k : int, number of clusters

    Returns
    -------
    centroids : list of list of numerics, length = k, dimensions same as X
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it
    """
    centroids = X[:k]
    for _ in range(iterations):
        centroids, centroid_assignments = update_centroids(centroids, X)
    return centroids, centroid_assignments


def update_centroids(centroids, X):
    """Helper function to move centroids based on new location.

    Parameters
    ----------
    centroids : list of list of numerics, length = k, dimensions same as X
    X         : list of lists of numerics, data in length of inner list
                dimensional space
    
    Returns
    -------
    centroids : list of list of numerics, length = k, dimensions same as X
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it
    """
    centroid_assignments = get_new_assignments(centroids, X)
    new_centroids = calculate_new_centroids(centroid_assignments)
    return new_centroids, centroid_assignments


def get_new_assignments(centroids, X):
    """Helper function to find closest centroid for each data point.

    Parameters
    ----------
    centroids : list of list of numerics, length = k, dimensions same as X
    X         : list of lists of numerics, data in length of inner list
                dimensional space
    
    Returns
    -------
    centroid_assignments: list of list of lists, for each centroid the data
                          points that are closest to it
    """
    centroid_assignments = [[] for _ in centroids]
    for x in X:
        closest_dist = 1e100
        closest_centroid = None
        for centroid_idx, centroid_location in enumerate(centroids):
            current_dist = list_euclidean_dist(centroid_location, x)
            if current_dist < closest_dist:
                closest_dist = current_dist
                closest_centroid = centroid_idx
        centroid_assignments[closest_centroid].append(x)
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
        centroid = [sum(dim)/len(dim) for dim in zip(*centroid_assignment)]
        new_centroids.append(centroid) 
    return new_centroids

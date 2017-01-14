def base_python(X, k):
    centroids = X[:k]
    for _ in range(1000):
        centroids, centroid_assignments = update_centroids(centroids, X)
    return centroids, centroid_assignments


def update_centroids(centroids, X):
    centroid_assignments = get_new_assignments(centroids, X)
    new_centroids = calcualte_new_centroids(centroid_assignments)
    return new_centroids, centroid_assignments


def get_new_assignments(centroids, X):
    centroid_assignments = [[] for _ in centroids]
    for x in X:
        closest_dist = 10000000000000
        closest_centroid = None
        for centroid_idx, centroid_location in enumerate(centroids):
            current_dist = list_euclidean_dist(centroid_location, x)
            if current_dist < closest_dist:
                closest_dist = current_dist
                closest_centroid = centroid_idx
        centroid_assignments[closest_centroid].append(x)
    return centroid_assignments


def list_euclidean_dist(a, b):
    return sum((da - db) ** 2 for da, db in zip(a, b)) ** 0.5


def calcualte_new_centroids(centroid_assignments):
    new_centroids = []
    for centroid_assignment in centroid_assignments:
        centroid = [sum(dim)/len(dim) for dim in zip(*centroid_assignment)]
        new_centroids.append(centroid) 
    return new_centroids

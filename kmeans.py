import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def base_python(X, k):
    centroids = [X[np.random.randint(0, len(X))] for _ in range(k)]
    for _ in range(1000):
        centroids = update_centroids(centroids, X)
    return centroids


def update_centroids(centroids, X):
    centroid_assignments = get_new_assignments(centroids, X)
    new_centroids = calcualte_new_centroids(centroid_assignments)
    return new_centroids


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


def plot_everything(X, centroids):
    plt.scatter(*X.T, c='b', alpha=0.5)
    plt.scatter(*zip(*centroids), c='r', s=30)
    plt.show()

if __name__ == '__main__':
    np.random.seed = 42
    X, y = make_classification(n_samples=100, n_features=2,
                               n_redundant=0, class_sep=2,
                               n_clusters_per_class=1)
    centroids = base_python(X.tolist(), k=2)
    plot_everything(X, centroids)

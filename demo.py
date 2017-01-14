import kmeans
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def plot_everything(centroids, assignments):
    colors = ('b', 'g', 'r')
    for centroid, assigns, color in zip(centroids, assignments, colors):
        plt.scatter(*zip(*assigns), c=color, alpha=0.35)
        plt.scatter(*centroid, c=color, marker='*', s=300)
    plt.show()


if __name__ == '__main__':
    np.random.seed = 42
    X, y = make_classification(n_samples=100, n_features=2,
                               n_redundant=0, class_sep=2,
                               n_clusters_per_class=1)
    centroids, assignments = kmeans.base_python(X.tolist(), k=2)
    plot_everything(centroids, assignments)

import argparse
import numpy as np
import kmeans
import matplotlib.pyplot as plt


def make_blobs(n_points=100):
    points = np.array([[0.5, 0.5], [-0.5, -0.5]])
    noise = np.random.normal(0, .25, size=(int(n_points/2), 2))
    noised_points = points[:, None] + noise
    return noised_points.reshape(-1, 2)


def plot_everything(centroids, assignments, legend=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colors = ('b', 'g', 'r')
    for centroid, assigns, color in zip(centroids, assignments, colors):
        ax.scatter(*zip(*assigns), c=color, alpha=0.35, s=30, label='blob')
        ax.scatter(*centroid, c=color, marker='*', s=300, label='center')
    if legend:
        plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run demo of numeric Python with k-means.')
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--numpy', action='store_true')
    args = parser.parse_args()

    X = make_blobs()
    if args.base:
        base_centroids, base_assignments = kmeans.base_python(X.tolist(), k=2)
        plot_everything(base_centroids, base_assignments)
    if args.numpy:
        numpy_centroids, numpy_assignments = kmeans.numpy(X, k=2)
        plot_everything(numpy_centroids, numpy_assignments)

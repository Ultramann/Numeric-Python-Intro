import argparse
import numpy as np
import kmeans
import matplotlib.pyplot as plt
from time import time


def make_blobs(n_points):
    points = np.array([[0.5, 0.5], [-0.5, -0.5]])
    noise = np.random.normal(0, .25, size=(int(n_points/2), 2))
    noised_points = points[:, None] + noise
    return noised_points.reshape(-1, 2)


def plot_setup(centroids, assignments, ax, title=None, legend=False):
    colors = ('b', 'g', 'r')
    for centroid, assigns, color in zip(centroids, assignments, colors):
        size_alpha = 1 / (np.log2(len(assignments)) + 2)
        ax.scatter(*zip(*assigns), c=color, alpha=size_alpha, s=30, label='blob')
        ax.scatter(*centroid, c=color, marker='*', s=300, label='center')
    if title:
        ax.set_title(title)
    if legend:
        plt.legend(loc='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run demo of numeric Python with k-means.')
    parser.add_argument('--count', default=100, type=int)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--numpy', action='store_true')
    parser.add_argument('--ex', action='store_true')
    args = parser.parse_args()

    X = make_blobs(args.count)
    if args.ex:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        numpy_centroids, numpy_assignments = kmeans.numpy(X, k=2)
        plot_setup(numpy_centroids, numpy_assignments, ax, 'Example k-means', True)
        plt.show()

    if args.base:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        base_centroids, base_assignments = kmeans.base_python(X.tolist(), k=2)
        plot_setup(base_centroids, base_assignments, ax, 'Base Python k-means')
        plt.show()

    if args.numpy:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        numpy_centroids, numpy_assignments = kmeans.numpy(X, k=2)
        plot_setup(numpy_centroids, numpy_assignments, ax, 'NumPy k-means')
        plt.show()

    if not any([args.ex, args.base, args.numpy]):
        fig, ax_lst = plt.subplots(1, 2, figsize=(16, 8))
        for ax, data, algo, km in zip(ax_lst, (X.tolist(), X), ('Base Python', 'NumPy'),
                                                    (kmeans.base_python, kmeans.numpy)):
            start_time = time()
            centroids, assignments = km(data, k=2)
            total_time = time() - start_time
            timed_title = '{}: {:.2f} seconds'.format(algo, total_time)
            plot_setup(centroids, assignments, ax, timed_title)
        fig.suptitle('Timing: {} Data Points'.format(args.count), fontsize=25)
        plt.show()

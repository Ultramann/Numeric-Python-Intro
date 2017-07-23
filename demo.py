from time import time
import argparse
import kmeans
import matplotlib.pyplot as plt
import numpy as np


def make_blobs(n_points):
    """Convenience function to create data in two clusters, one around
    (0.5, 0,5) and the other (-0.5, -0.5).

    Parameters
    ----------
    n_points : int, total number of data points in both clusters

    Returns
    -------
    ndarray, 2D
    """
    points = np.array([[0.5, 0.5], [-0.5, -0.5]])
    noise = np.random.normal(0, .25, size=(int(n_points/2), 2))
    noised_points = points[:, None] + noise
    return noised_points.reshape(-1, 2)


def plot_setup(centroids, assignments, ax, title=None, legend=False):
    """Convenience function to plot clusters and centroids.

    Parameters
    ----------
    centroids : list-like 2D
    assignments : list of list-like
    ax : matplotlib axis object
    title : str, to put on ax
    legend : bool, plot legend
    """
    for centroid, assigns, color in zip(centroids, assignments, 'br'):
        size_alpha = 1 / (np.log2(len(assignments)) + 2)
        ax.scatter(*zip(*assigns), c=color, alpha=size_alpha, s=30, label='blob')
        ax.scatter(*centroid, c='k', marker='*', alpha=0.7, s=300, label='center')
    if title:
        ax.set_title(title, fontsize=15)
    if legend:
        plt.legend(loc='best')


def plot_single(X, km, label, legend=False):
    """Cluster and plot results for single algorithm.

    Parameters
    ----------
    X : list-like 2D
    km : function, kmeans clustering function
    label : str, for plot
    legend : bool, plot legend
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    centroids, assignments = km(X, k=2)
    plot_setup(centroids, assignments, ax, label, legend)
    plt.show()


def plot_timing_comp():
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    nums = [10, 100, 1000, 2500, 5000]
    times = ([], [])
    kms = (kmeans.base_python, kmeans.numpy)
    for num_points in nums:
        X = make_blobs(num_points)
        for data, km, km_time in zip((X.tolist(), X), kms, times):
            start_time = time()
            km(data, k=2)
            km_time.append(time() - start_time)

    ax.plot(nums, times[0], c='r', label='Base')
    ax.plot(nums, times[1], c='b', label='NumPy')
    ax.set_xlabel('Number of Points to Cluster', fontsize=13)
    ax.set_ylabel('Time to Cluster - Seconds', fontsize=13)
    ax.legend(loc='best')
    ax.set_title('Time Scaling for 1000 Iterations', fontsize=15)
    plt.show()


def plot_cluster_comp(X):
    fig, ax_lst = plt.subplots(1, 2, figsize=(16, 8))
    params = zip(ax_lst, (X.tolist(), X), ('Base Python', 'NumPy'),
                                (kmeans.base_python, kmeans.numpy))
    for ax, data, algo, km in params:
        start_time = time()
        centroids, assignments = km(data, k=2)
        total_time = time() - start_time
        timed_title = '{}: {:.2f} seconds'.format(algo, total_time)
        plot_setup(centroids, assignments, ax, timed_title)
    fig.suptitle('Timing - {} Data Points'.format(args.count), fontsize=25)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run demo of numeric Python with k-means.')
    parser.add_argument('--count', default=100, type=int,
                        help='number of data points to generate in total')
    parser.add_argument('--base', action='store_true',
                        help='run with base python implementation')
    parser.add_argument('--numpy', action='store_true',
                        help='run with numpy implementation')
    parser.add_argument('--comp', action='store_true',
                        help='run to create time comparison plot')
    parser.add_argument('--ex', action='store_true',
                        help='run to create example plot')
    args = parser.parse_args()

    if args.comp:
        plot_timing_comp()
            
    X = make_blobs(args.count)

    if args.ex:
        plot_single(X, kmeans.numpy, 'Example k-means', legend=True)

    if args.base:
        plot_single(X, kmeans.base_python, 'Base Python k-means')

    if args.numpy:
        plot_single(X, kmeans.numpy, 'NumPy k-means')

    if not any([args.comp, args.ex, args.base, args.numpy]):
        plot_cluster_comp(X)

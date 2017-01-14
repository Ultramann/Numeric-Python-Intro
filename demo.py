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
    X = make_blobs()
    centroids, assignments = kmeans.base_python(X.tolist(), k=2)
    plot_everything(centroids, assignments)

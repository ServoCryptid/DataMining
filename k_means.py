import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


class MyKMeans:
    """
    Class that implements the k-means algorithm for clustering data
    """

    def plot_data(self):
        plt.scatter(self.dataset, np.zeros(len(self.dataset)))
        plt.scatter(self.centroids, np.zeros(len(self.centroids)), marker='p', s=100,
                    c='r', label='centroid')
        plt.show()

    def compute_centroids(self, labels, dataset, clusters):
        self.centroids = np.zeros((self.clusters_number, dataset.shape[1]))
        for cluster in range(clusters):#the centeroid is the mean of all the values that are in that cluster
            self.centroids[cluster, :] = np.mean(dataset[labels == cluster, :], axis=0)

    def find_closest_cluster(self, distances):
        """
        Method that find the minimum distance for each data point, one point /column
        :param distances:
        :return: minimum distance for each data point
        """
        return np.argmin(distances, axis=0)

    def compute_distances(self):
        distances = np.zeros((self.centroids.shape[0], self.dataset.shape[0]))

        for cluster in range(self.centroids.shape[0]):
            distances[cluster, :] = np.square(norm(dataset - self.centroids[cluster, :], axis=1))

        return distances

    def initialize_centroids(self):
        """
        Shuffles the dataset and select randomly the values for the centroids
        """
        shuffled_dataset = np.random.permutation(self.dataset.shape[0])  # the number of rows

        return dataset[shuffled_dataset[:clusters]]

    def train(self, iterations=100):
        """
        Function that implements the kmeans clustering algorithm
        :param clusters: number of iterations, by default is 100
        :return: the values of the clusters computed
        """

        for i in range(iterations):
            old_centroids = self.centroids
            distances = self.compute_distances()
            labels = self.find_closest_cluster(distances)
            print(labels)
            self.compute_centroids(labels, dataset, clusters)
            if np.all(old_centroids == self.centroids):  # if not even a single centroid has changed
                break

        return self.centroids

    def __init__(self, clusters, dataset):  #by default it's 100 iterations
        self.clusters_number = clusters
        self.dataset = dataset
        self.centroids = self.initialize_centroids()


if __name__ == "__main__":
    dataset = np.array([[2, 4, 3, 4, 60, 70, 80], [1, 2, 100, 2054, 30, 2600, 9504]])
    dataset = dataset.transpose()
    clusters = 2
    cluster_algo = MyKMeans(clusters, dataset)

    print(f"Final centroids: {cluster_algo.train()}")

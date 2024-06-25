import numpy as np
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN


# get the two points furthest apart in the skeleton
def get_furthest_points(skeleton):
    # Get the coordinates of the skeleton
    coords = np.argwhere(skeleton)

    # Calculate the pairwise distances
    distances = np.linalg.norm(coords[:, None] - coords, axis=-1)

    # Get the indices of the maximum distance
    idx = np.unravel_index(np.argmax(distances), distances.shape)

    # Get the two points
    point1 = coords[idx[0]]
    point2 = coords[idx[1]]

    return point1, point2


def get_first_and_last_point(skeleton):
    # Get the coordinates of the skeleton
    coords = np.argwhere(skeleton)

    return coords[0], coords[-1]


def first_n_last_conv(skeleton):
    pass


# Endpoints are defined as pixels that have only one neighbor so the convolutional result for that pixel would be itsles (the value at the center of the kernel) + 1 (for the neighbor)
def get_endpoints(skeleton):
    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The endpoints are where the convolved image is 11
    endpoints = np.argwhere((convolved == 11))

    return endpoints


def get_branching_points(skeleton):
    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The branching points are where the convolved image is 30 or more
    branching_points = np.argwhere(convolved == 13)

    # To keep only one branching point per branch, you can use a clustering algorithm to group neighboring branching points together and then keep only one point from each cluster. A simple and efficient algorithm for this task is DBSCAN (Density-Based Spatial Clustering of Applications with Noise) from the sklearn library.

    # DBSCAN groups together points that are packed closely together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions. The number of clusters, hence the number of branching points, is determined by the input parameters.

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=2, min_samples=1).fit(branching_points)

    # Get the coordinates of the cluster centers
    cluster_centers = []
    for cluster_id in set(clustering.labels_):
        cluster_points = branching_points[clustering.labels_ == cluster_id]
        cluster_center = cluster_points.mean(axis=0)
        cluster_centers.append(cluster_center)

    return np.array(cluster_centers)

    # return branching_points


# make the kernel large not 8 point connected but 24 point connected so we can look further away
def get_crossing_points(skeleton):
    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The crossing points are where the convolved image is 40 or more
    crossing_points = np.argwhere(convolved >= 14)

    return crossing_points

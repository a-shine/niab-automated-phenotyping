"""
Utility function to extract the endpoints, branching points, and crossing
points from a skeletonized image.
"""

import numpy as np
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN


def get_furthest_points(skeleton):
    """
    Get the two points that are furthest apart in the skeleton.

    Args:
        skeleton (np.ndarray): The skeletonized image.

    Returns:
        tuple: The coordinates of the two points that are furthest apart.
    """
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
    """
    Get the first and last point of the skeleton.

    Args:
        skeleton (np.ndarray): The skeletonized image.

    Returns:
        tuple: The coordinates of the first and last point.
    """
    # Get the coordinates of the skeleton
    coords = np.argwhere(skeleton)

    return coords[0], coords[-1]


def get_endpoints(skeleton):
    """
    Get the endpoints of the skeleton by convolving it with a kernel that
    highlights the endpoints.

    Endpoints are pixels that have only one neighbor in the skeleton.

    Args:
        skeleton (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The coordinates of the endpoints.
    """

    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The endpoints are where the convolved image is 11
    endpoints = np.argwhere((convolved == 11))

    return endpoints


def get_branching_points(skeleton):
    """
    Get the branching points of the skeleton by convolving it with a kernel
    that highlights the branching points.

    Branching points are pixels that have three or more neighbors in the
    skeleton.

    Args:
        skeleton (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The coordinates of the branching points.
    """
    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The branching points are where the convolved image is 30 or more
    branching_points = np.argwhere(convolved == 13)

    # To keep only one branching point per branch, use a clustering algorithm
    # to group neighboring branching points together and then keep only one
    # point from each cluster using DBSCAN (Density-Based Spatial Clustering of
    # Applications with Noise) algorithm.

    # DBSCAN groups together points that are packed closely together (points
    # with many nearby neighbors), marking as outliers points that lie alone in
    # low-density regions. The number of clusters, hence the number of
    # branching points, is determined by the input parameters.

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=2, min_samples=1).fit(branching_points)

    # Get the coordinates of the cluster centers
    cluster_centers = []
    for cluster_id in set(clustering.labels_):
        cluster_points = branching_points[clustering.labels_ == cluster_id]
        cluster_center = cluster_points.mean(axis=0)
        cluster_centers.append(cluster_center)

    return np.array(cluster_centers)


def get_crossing_points(skeleton):
    """
    Get the crossing points of the skeleton by convolving it with a kernel
    that highlights the crossing points.

    Crossing points are pixels that have four or more neighbors in the
    skeleton.

    Args:
        skeleton (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The coordinates of the crossing points.
    """

    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode="constant", cval=0)

    # The crossing points are where the convolved image is 40 or more
    crossing_points = np.argwhere(convolved >= 14)

    return crossing_points

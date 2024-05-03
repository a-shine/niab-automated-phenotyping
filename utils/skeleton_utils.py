import numpy as np
from scipy.ndimage import convolve

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
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode='constant', cval=0)

    # visialize the convolved image
    # plt.imshow(convolved, cmap='gray')

    # The endpoints are where the convolved image is 11
    endpoints = np.argwhere((convolved == 11))

    # plt.scatter(endpoints[:, 1], endpoints[:, 0], color='red', s=10)

    return endpoints

def get_branching_points(skeleton):
    # Define the convolutional kernel
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    # Apply the kernel to the skeleton
    convolved = convolve(skeleton, kernel, mode='constant', cval=0)

    # The branching points are where the convolved image is 30 or more
    branching_points = np.argwhere(convolved >= 13)

    return branching_points
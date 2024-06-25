import glob

import cv2
import numpy as np


def load_image_ts(dataset_root: str, exp: int, block: int, pot: int) -> list[cv2.Mat]:
    """
    Load a time series of images from a specified experiment, block, and pot.

    Args:
        dataset_root (str): The root directory where the dataset is stored.
        experiment_number (int): The number of the experiment.
        block_number (int): The number of the block.
        pot_number (int): The number of the pot.

    Returns:
        list: A list of images loaded with OpenCV.

    """
    # Define the base path for the images
    base_path = f"{dataset_root}/EXP{exp:02d}/Top_Images/Top_Images_Clean_Rename/EXP{exp:02d}_Block{block:02d}/"

    # Use glob to get all the image file paths
    image_paths = glob.glob(
        base_path
        + f"EXP{exp:02d}_Block{block:02d}_Rename*/Exp{exp:02d}_Block{block:02d}_Image*_Pot{pot:03d}.jpg"
    )

    # Sort the paths to ensure they are in the correct order
    image_paths.sort()

    # Load the images into OpenCV
    images = [cv2.imread(path) for path in image_paths]

    return images


def reduce_resolution(image: cv2.Mat, scale_percent: int) -> cv2.Mat:
    """
    Reduce the resolution of an image by a certain percentage.

    Args:
        image (cv2.Mat): The original image.
        scale_percent (int): The percentage of the original size to which the image should be scaled.
                             For example, if scale_percent is 50, the image will be half its original size.

    Returns:
        cv2.Mat: The image after scaling.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def white_balance(image: cv2.Mat) -> cv2.Mat:
    """
    Apply automatic white balancing to an image using the Gray World assumption.

    This function is an implementation of the method described in the following StackOverflow post:
    https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574

    Args:
        img (cv2.Mat): The original BGR image.

    Returns:
        cv2.Mat: The image after white balancing.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

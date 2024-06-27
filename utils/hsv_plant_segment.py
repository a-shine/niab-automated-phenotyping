import cv2
import numpy as np

import utils.image_utils as wb


def segment_plants(img, hsvmin_wb, hsvmax_wb):
    """
    Segment plants in the image based on the HSV thresholds.

    Args:
        img: The input image.
        hsvmin_wb: The minimum HSV values for the white balanced image.
        hsvmax_wb: The maximum HSV values for the white balanced image.

    Returns:
        wb_img: The white balanced image.
        output: The segmented image.
    """
    lower_wb = np.array(hsvmin_wb)
    upper_wb = np.array(hsvmax_wb)

    # Create HSV Image from white balanced image and threshold into a range.
    wb_img = wb(img)
    hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
    mask_wb = cv2.inRange(hsv_img, lower_wb, upper_wb)

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask_wb, cv2.MORPH_OPEN, kernel)
    output = cv2.dilate(opening, kernel, iterations=10)

    return wb_img, output

"""
The script loads the images from the dataset directory and applies the HSV
segmentation to the images. The script uses the segment_plants function to
segment the plants in the images based on the specified HSV thresholds.

The script saves the white balanced images and the segmentation masks to the
output folder.

The script uses the following parameters:
- HSVMIN_WB: The minimum HSV values for the white balanced image
- HSVMAX_WB: The maximum HSV values for the white balanced image

Example:
    python hsv_plant_segment.py

Note: The script assumes that the dataset directory contains images in the same
format as the NIAB dataset. Make sure to use the correct HSV thresholds for
segmentation of white balanced images.
"""

import glob
import os

import cv2
import numpy as np

from utils.image_utils import white_balance as wb

HSVMIN_WB = (30, 80, 0)
HSVMAX_WB = (65, 255, 255)


def segment_plants(img):
    lower_wb = np.array(HSVMIN_WB)
    upper_wb = np.array(HSVMAX_WB)

    # Create HSV Image from white balanced image and threshold into a range.
    wb_img = wb(img)
    hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
    mask_wb = cv2.inRange(hsv_img, lower_wb, upper_wb)

    # Combine the masks
    # mask = cv2.bitwise_or(mask_raw, mask_wb)

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask_wb, cv2.MORPH_OPEN, kernel)
    output = cv2.dilate(opening, kernel, iterations=10)

    return wb_img, output


if __name__ == "__main__":
    # Get all the images in the folder and subfolders of ./datasets/niab
    images = glob.glob(
        "./datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename/EXP01_Block01/EXP01_Block01_Rename07_20201206/*.jpg",
        recursive=True,
    )

    # Create the output folder
    if not os.path.exists("./output"):
        os.makedirs("./output")

        # make a img and mask folder
        os.makedirs("./output/img")
        os.makedirs("./output/mask")

    # Loop through all the images in th
    for i in range(len(images)):
        img = cv2.imread(images[i])
        wb_img, mask = segment_plants(img)

        # Save the images and masks
        cv2.imwrite("./output/imgs/" + os.path.basename(images[i]), wb_img)
        cv2.imwrite("./output/masks/" + os.path.basename(images[i]), mask)

        print("Processed: " + str(i) + " of " + str(len(images)) + " images")

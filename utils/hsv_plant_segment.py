import cv2
import numpy as np
import utils.white_balance as wb

HSVMIN_RAW = (30, 170, 0)
HSVMAX_RAW = (65, 255, 255)
HSVMIN_WB = (30, 80, 0)
HSVMAX_WB = (65,255, 255)

def segment_plants(img):
    # lower_raw = np.array(HSVMIN_RAW)
    # upper_raw = np.array(HSVMAX_RAW)
    lower_wb = np.array(HSVMIN_WB)
    upper_wb = np.array(HSVMAX_WB)

    # Create HSV Image and threshold into a range.
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask_raw = cv2.inRange(hsv_img, lower_raw, upper_raw)

    # Create HSV Image from white balanced image and threshold into a range.
    wb_img = wb.white_balance(img)
    hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
    mask_wb = cv2.inRange(hsv_img, lower_wb, upper_wb)

    # Combine the masks
    # mask = cv2.bitwise_or(mask_raw, mask_wb)

    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask_wb, cv2.MORPH_OPEN, kernel)
    output = cv2.dilate(opening, kernel, iterations=10)

    return wb_img, output
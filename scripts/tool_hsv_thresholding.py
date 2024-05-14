# GUI tool to help operator to find the HSV threshold values for HSV 
# segmentation suited to a specified image.


import cv2
import numpy as np

def nothing(x):
    pass

# Load your image
img = cv2.imread('./datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename/EXP01_Block01/EXP01_Block01_Rename06_20201204/Exp01_Block01_Image06_Pot034.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing)
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

while(1):
    # grab the frame
    

    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')
    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Display the resulting frame
    cv2.imshow('image', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
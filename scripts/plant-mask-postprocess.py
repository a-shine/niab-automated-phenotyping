

# read image
img = cv2.imread("./data/test2/Exp01_Block01_Image04_Pot001_mask.png")

# define the structure for morphological operations
kernel = np.ones((23,23),np.uint8)

# perform morphological operations
# erosion followed by dilation (opening)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# plot image
cv2.imshow("image", opening)

# wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
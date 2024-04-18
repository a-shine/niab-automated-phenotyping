import cv2
import numpy as np

# Load the image and the mask
# image = cv2.imread('./output/img/Exp01_Block01_Image01_Pot001.jpg')
mask = cv2.imread('./output/mask/Exp01_Block01_Image01_Pot001.jpg', cv2.IMREAD_GRAYSCALE)

# Downscale the images
# scale_percent = 20  # percent of original size
# width = int(mask.shape[1] * scale_percent / 100)
# height = int(mask.shape[0] * scale_percent / 100)
# dim = (width, height)
# # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)


# Create a window
cv2.namedWindow('image')

# Create a trackbar for brush size
cv2.createTrackbar('Size', 'image', 5, 50, lambda x: None)

# Create a canvas to draw on
canvas = mask.copy()
canvas_display = canvas.copy()

# Mouse callback function
def draw(event, x, y, flags, param):
    global canvas_display
    canvas_display = canvas.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the brush size from the trackbar
        size = cv2.getTrackbarPos('Size', 'image')
        # Draw a circle on the display canvas
        cv2.circle(canvas_display, (x, y), size, (255), 1)
        # FIXME: You really have to click and then drag while clicking for it to work
        if flags == cv2.EVENT_FLAG_LBUTTON:
            print(f'Brush clicked at x: {x}, y: {y}')
            # Draw a black circle on the canvas
            cv2.circle(canvas, (x, y), size, (0), -1)

# Set the mouse callback function
cv2.setMouseCallback('image', draw)

while True:
    # Display the image and the mask side by side
    # display = np.hstack((image, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)))
    display = cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', display)

    # If 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
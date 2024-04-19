import cv2
import numpy as np
import os

# Get a list of all the mask files
mask_dir = './output/mask'
img_dir = './output/img'
mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')]
mask_files.sort()
print(f'Found {len(mask_files)} mask files')
current_index = 136

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
cv2.createTrackbar('Size', 'image', 20, 200, lambda x: None)

# Mouse callback function
def draw(event, x, y, flags, param):
    global canvas, canvas_display
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

# remove the mask files that are already edited i.e. before the current_index
mask_files = mask_files[current_index:]

for mask_file in mask_files:
    print(f'Editing mask: {mask_file}')
    # Load the mask
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    canvas = mask.copy()
    canvas_display = canvas.copy()

    # Display the img and mask with the mask overlayed with low opacity
    img_file = os.path.join(img_dir, os.path.basename(mask_file))
    img = cv2.imread(img_file)
    
    while True:
        combined = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow('image', combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            cv2.imwrite(mask_file, canvas)
            print(f'Saved mask and moving to next mask')
            print(f'Current index: {current_index} out of {len(mask_files)}')
            break
        elif key == ord('s'):
            # Save the mask
            cv2.imwrite(mask_file, canvas)
            print(f'Saved mask')

cv2.destroyAllWindows()
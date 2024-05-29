import cv2
import numpy as np
import os

# Define the drawing state and color
drawing = False
class_id = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
ix, iy = -1, -1
min_line_width = 2
max_line_width = 4
output_dir = 'output'
img_count = 0

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create output/binary_masks directory
if not os.path.exists(os.path.join(output_dir, 'binary_masks')):
    os.makedirs(os.path.join(output_dir, 'binary_masks'))

    # create instance_masks directory
if not os.path.exists(os.path.join(output_dir, 'instance_masks')):
    os.makedirs(os.path.join(output_dir, 'instance_masks'))

# Create a black image for the binary mask
binary_mask = np.zeros((345, 460), np.uint8)

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing, img, binary_mask, class_id, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_width = np.random.randint(min_line_width, max_line_width)
            cv2.line(img, (ix, iy), (x, y), colors[class_id], line_width)
            cv2.line(binary_mask, (ix, iy), (x, y), 255, line_width)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_width = np.random.randint(min_line_width, max_line_width)
        cv2.line(img, (ix, iy), (x, y), colors[class_id], line_width)
        cv2.line(binary_mask, (ix, iy), (x, y), 255, line_width)

# Create a black image and a window
img = np.zeros((345, 460, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('n'):  # Press 'n' to change the class
        class_id = (class_id + 1) % len(colors)
    elif k == ord('t'):  # Press 't' to save the current image and create a new one
        cv2.imwrite(os.path.join(output_dir, f'instance_masks/{img_count}.jpg'), img)
        cv2.imwrite(os.path.join(output_dir, f'binary_masks/{img_count}.jpg'), binary_mask)
        img = np.zeros((345, 460, 3), np.uint8)
        binary_mask = np.zeros((345, 460), np.uint8)
        img_count += 1
    elif k == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
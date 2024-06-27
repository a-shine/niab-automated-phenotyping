"""
Graphical tool to create dummy semantic instance segmentation dataset by
drawing lines on a black image similar to shoots.

The script creates a window with a black image and allows the user to draw lines
on the image using the mouse. The script saves the binary and instance masks
to the output directory.

The script uses the following parameters:
- min_line_width: The minimum width of the line to draw
- max_line_width: The maximum width of the line to draw
- output_dir: The output directory to save the masks
- auto_increment: Flag to toggle auto-incrementing the class ID

Example:
    python tool_dummy_semantic_to_instance_drawing.py

Note: The script creates a window with a black image and allows the user to draw
lines on the image using the mouse. Press 'n' to save the current image and create
a new one. Press 'i' to change the class ID. Press 'a' to toggle auto-incrementing
the class ID.
"""

import os

import cv2
import numpy as np

# Define the drawing state and color
drawing = False
class_id = 1
ix, iy = -1, -1
min_line_width = 2
max_line_width = 4
output_dir = "output"
img_count = 0
auto_increment = True

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create output/binary_masks directory
if not os.path.exists(os.path.join(output_dir, "binary_masks")):
    os.makedirs(os.path.join(output_dir, "binary_masks"))

# create instance_masks directory
if not os.path.exists(os.path.join(output_dir, "instance_masks")):
    os.makedirs(os.path.join(output_dir, "instance_masks"))

# Create a black image for the binary mask
binary_mask = np.zeros((345, 460), np.uint8)


# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing, instance_mask, binary_mask, class_id, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        if auto_increment:
            class_id += 1

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_width = np.random.randint(min_line_width, max_line_width)
            cv2.line(instance_mask, (ix, iy), (x, y), class_id, line_width)
            cv2.line(binary_mask, (ix, iy), (x, y), 255, line_width)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_width = np.random.randint(min_line_width, max_line_width)
        cv2.line(instance_mask, (ix, iy), (x, y), class_id, line_width)
        cv2.line(binary_mask, (ix, iy), (x, y), 255, line_width)


# Create a black image and a window
instance_mask = np.zeros((345, 460), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_line)

while 1:
    # Normalize the grayscale values to the full range of 0-255
    img_normalized = cv2.normalize(
        instance_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    # Apply the rainbow colormap to the normalized image
    display_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
    cv2.imshow("image", display_image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("i"):  # Press 'n' to change the class
        class_id += 1
    elif k == ord("a"):  # Press 'a' to toggle auto increment
        auto_increment = not auto_increment
        print(f"Auto increment: {auto_increment}")
    elif k == ord("n"):  # Press 't' to save the current image and create a new one
        class_id = 1
        cv2.imwrite(
            os.path.join(output_dir, f"binary_masks/{img_count}.png"), binary_mask
        )
        cv2.imwrite(
            os.path.join(output_dir, f"instance_masks/{img_count}.png"), instance_mask
        )
        instance_mask = np.zeros((345, 460), np.uint8)
        binary_mask = np.zeros((345, 460), np.uint8)
        img_count += 1
    elif k == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()

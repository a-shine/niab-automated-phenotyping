# GUI tool to help labeler correct image masks. This is meant to be used as
# part of a semi-automatic labeling pipeline where the bulk of the segment
# labeling is carried out using a naive approach such as HSV thresholding and
# then the operator corrects the masks using this tool before feeding the
# corrected masks to the neural network for training.

import os

import cv2

# Get a list of all the mask files
mask_dir = "./output/mask"
img_dir = "./output/img"
mask_files = [
    os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".jpg")
]
mask_files.sort()
print(f"Found {len(mask_files)} mask files")
total_files = len(mask_files)
current_index = 0

# Create a window
cv2.namedWindow("image")

# Create a trackbar for brush size
cv2.createTrackbar("Size", "image", 20, 200, lambda x: None)


# Mouse callback function
def draw_or_erase(event, x, y, flags, param):
    global canvas, canvas_display
    canvas_display = canvas.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the brush size from the trackbar
        size = cv2.getTrackbarPos("Size", "image")
        # Draw a circle on the display canvas
        cv2.circle(canvas_display, (x, y), size, (255), 1)
        # FIXME: You really have to click and then drag while clicking for it to work
        if flags == cv2.EVENT_FLAG_RBUTTON:
            print(f"Eraser clicked at x: {x}, y: {y}")
            cv2.circle(canvas, (x, y), size, (0), -1)
        if flags == cv2.EVENT_FLAG_LBUTTON:
            print(f"Brush clicked at x: {x}, y: {y}")
            cv2.circle(canvas, (x, y), size, (255), -1)


# Set the mouse callback function
cv2.setMouseCallback("image", draw_or_erase)

# remove the mask files that are already edited i.e. before the current_index
mask_files = mask_files[current_index:]

for mask_file in mask_files:
    print(f"Editing mask: {mask_file}")
    # Load the mask
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    canvas = mask.copy()
    canvas_display = canvas.copy()

    # Display the img and mask with the mask overlayed with low opacity
    img_file = os.path.join(img_dir, os.path.basename(mask_file))
    img = cv2.imread(img_file)

    while True:
        combined = cv2.addWeighted(
            img, 0.5, cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR), 0.5, 0
        )
        cv2.imshow("image", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            cv2.imwrite(mask_file, canvas)
            print("Saved mask and moving to next mask")
            print(f"Current index: {current_index} out of {total_files}")
            current_index += 1
            break
        elif key == ord("s"):
            # Save the mask
            cv2.imwrite(mask_file, canvas)
            print("Saved mask")

cv2.destroyAllWindows()

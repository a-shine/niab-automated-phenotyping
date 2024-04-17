import cv2
from rembg import remove
import argparse
import os


def segment_image(input_path, output_path):
    # create output directory if it doesn't exist
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    input_img = cv2.imread(input_path)
    output_img = remove(input_img)
    # Crop out the bottom part of the image as it has artifacts from the lab apparatus
    output_img = output_img[0:output_img.shape[0] - 500, :]
    # Crop out the top part of the image as it has artifacts from the lab apparatus
    output_img = output_img[200:output_img.shape[0], :]
    # Crop the left part of the image as it has artifacts from the lab apparatus
    output_img = output_img[:, 900:output_img.shape[1]]
    # Crop the right part of the image as it has artifacts from the lab apparatus
    output_img = output_img[:, 0:output_img.shape[1] - 700]
    cv2.imwrite(output_path, output_img)


def segment_directory(input_dir, output_dir):
    # Get or create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Segmenting images in {input_dir} to {output_dir}")
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.jpg', '_removed.png'))
            segment_image(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment images.')
    parser.add_argument('--input', help='Input image file or directory.')
    parser.add_argument('--output', help='Output image file or directory.')
    # TODO: Add a verbose flag to the script
    args = parser.parse_args()

    if os.path.isdir(args.input):
        print("Segmenting images in directory")
        segment_directory(args.input, args.output)
    else:
        print("Segmenting single image")
        segment_image(args.input, "./segmented.png")

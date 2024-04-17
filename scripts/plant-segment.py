#!/usr/bin/python3

import argparse
import os
import csv
import cv2
import numpy as np
import glob
import colorsys

# Determines whether string is a directory or not
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# Finds plant segments of a specific colour
def generate_segments(in_path, recursive, hsvmin, hsvmax):

    wait_time = 1
    allimgs = glob.glob(os.path.join(in_path,"*.jpg"),recursive=recursive)
    
    for img in allimgs:
  
        dname = os.path.dirname(img)
        fname = os.path.basename(img)
        nname = os.path.splitext(fname)[0]+".png"
        mnname = os.path.splitext(fname)[0]+"_mask.png"
        nname_csv = os.path.splitext(fname)[0]+".csv"

        full = os.path.join(dname, nname)
        full_mask = os.path.join(dname, mnname)
        full_csv = os.path.join(dname, nname_csv)

        if os.path.isfile(full):
            print("Skipping", full)
            continue

        print("Loading ",img)

        image = cv2.imread(img)
        #image = white_balance(image)
        #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        orig = image.copy()

        # Set minimum and max HSV values to display
        lower = np.array(hsvmin)
        upper = np.array(hsvmax)

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        kernel = np.ones((3,3), np.uint8)
        output = cv2.dilate(mask, kernel, iterations=10)
        cv2.imwrite(full_mask, output)
 
        hei, wid = output.shape
        allrects = []

        for y in range(hei):
            for x in range(wid):
                if (output[y][x] == np.array([ 255, 255, 255])).all() :
                    print("flood")
                    seed = (x, y)
                    tupn = (100, 100, 100)
                    retval, image, mask, rect = cv2.floodFill(output, None, seedPoint=seed, newVal=tupn)
                    if rect[2] * rect[3] > 40 * 40:
                        cv2.rectangle(orig,rect,(0,255,0),3)
                        allrects.append(rect)

        with open(full_csv, 'w') as csvfile:
            fieldnames = ['x', 'y', 'width', 'height']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for rect in allrects:
                writer.writerow({'x': rect[0], 'y': rect[1], 'width': rect[2], 'height': rect[3]})

        cv2.imshow('image', orig)
        cv2.imwrite(full, orig)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Interface for segmenting plants")
    parser.add_argument("--hsvmin", dest="hsvmin", type=int, help="HSV min", nargs=3, required=True)
    parser.add_argument("--hsvmax", dest="hsvmax", type=int, help="HSV max", nargs=3, required=True)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=dir_path,
        help="Filename to load images from and to write output to",
        required=True
    )
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    
    args = parser.parse_args()
    generate_segments(args.in_path, args.recursive, args.hsvmin, args.hsvmax)





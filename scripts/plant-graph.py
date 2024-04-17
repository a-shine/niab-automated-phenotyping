#!/usr/bin/python3

import argparse
import os
import csv
import cv2
import re
import math
import numpy as np
import glob
import colorsys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Determines whether string is a directory or not
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def collapse(data):
    newdata = data

    def merge(d):
        mergeit = False

        for v1 in d:
            smallest = math.inf
            found = None

            for v2 in d:
                if v1 == v2:
                    continue
                euc = math.sqrt((v1['midx'] - v2['midx'])**2 + (v1['midy'] - v2['midy'])**2) #+ (v1['size'] - v2['size'])**2)
                if euc < 600:
                    found = v2
                    break

            if found is not None:
                # merge v1 and found
                d.remove(v1)
                d.remove(found)

                x = min(v1['x'], found['x'])
                y = min(v1['y'], found['y'])
                x_ = max(v1['x']+v1['width'], found['x']+found['width'])
                y_ = max(v1['y']+v1['height'], found['y']+found['height'])
                w = x_ - x
                h = y_ - y
                newel = {'x': x, 'y': y, 'size': w*h,'width': w, 'height':h,
                        'midx': x+int(w/2), 'midy': y+int(h/2)}

                # add new element that merges two
                d.append(newel)
                mergeit = True
                break
                
        if mergeit:
            return merge(d)
        else:
            return d 

    return merge(newdata)

def match(data1, data2):
    nop  = 1
    newl = []

    for v1 in data1:
        smallest = math.inf
        found = {}
        for v2 in data2:
            euc = math.sqrt((v1['midx'] - v2['midx'])**2 + (v1['midy'] - v2['midy'])**2) #+ (v1['size'] - v2['size'])**2)
            if euc < smallest:
                smallest = euc
                found = v2.copy()
                found['idv'] = v1['idv']
        if smallest < 300:
            newl.append(found)

    return newl

# Generate graph
def generate_graph(in_path, recursive):

    wait_time = 1
    allcsvs = glob.glob(os.path.join(in_path,"*.csv"),recursive=recursive)
    pot = {}
    for csvf in allcsvs:

        filen = os.path.basename(csvf)
        p = re.compile('Exp(\d+)_Block(\d+)_Image(\d+)_Pot(\d+).csv')
        m = p.match(filen)

        if int(m.group(4)) not in pot:
            pot[int(m.group(4))] = {}

        pot[int(m.group(4))][int(m.group(3))] = {'data':[],'image':csvf.replace('.csv','.jpg')}

        # Load data from CSVs and group by potid then imageid
        with open(csvf) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nrow = {}
                for k, v in row.items():
                    nrow[k] = int(v)

                if nrow['width']*nrow['height'] < 2000:
                    continue
                #         pot id         imageid
                pot[int(m.group(4))][int(m.group(3))]['data'].append({'x': nrow['x'], 'y': nrow['y'], 'size': nrow['width']*nrow['height'],'width': nrow['width'], 'height':nrow['height'],
                                                                      'midx': nrow['x']+int(nrow['width']/2), 'midy': nrow['y']+int(nrow['height']/2),
                                                                        })
        pot[int(m.group(4))][int(m.group(3))]['data'] = collapse(pot[int(m.group(4))][int(m.group(3))]['data'])

    # Search through all potential pots
    for potid in pot:
        found = None
        alldat = []
        first = None

        for imgid in sorted(pot[potid].keys(),reverse=True):
            if first is None:
                first = pot[potid][imgid]['image']
            if imgid - 1 >= 1:
                if found is None:
                    count = 1
                    for dat in pot[potid][imgid]['data']:
                        dat['idv'] = count
                        count += 1
                    alldat.append(pot[potid][imgid]['data'])
                    found = match(pot[potid][imgid]['data'], pot[potid][imgid-1]['data'])
                    alldat.append(found)
                else:
                    found = match(found, pot[potid][imgid-1]['data'])
                    alldat.append(found)
            nop = 1
    
        # remove
        potential = []
        for d in alldat[0]:
            potential.append(d['idv'])

        blank_image = np.zeros((3456,4608,3), np.uint8)
        blank_image = cv2.imread(first)
        lastd = {} 

        with open('graph-%d.csv' % potid, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=potential)
            c = 0
            writer.writeheader()
            rev = alldat[::-1]

            graphit = defaultdict(list)

            for l in rev:
                c += 1
                newdict = {}
                for v in l:
                    if v['idv'] in potential:

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        if v['idv'] in lastd:
                            cv2.line(blank_image,lastd[v['idv']],(v['x']+int(v['width']/2),v['y']+int(v['height']/2)),(255,0,0),5)
                        if c == len(rev):
                            cv2.rectangle(blank_image,(v['x'],v['y'],v['width'],v['height']) ,(0,0,255), 5)
                            cv2.putText(blank_image, "(%d,#%d)" % (c,v['idv']), (v['x'],v['y']), font, 2, (255, 0, 0), 6, cv2.LINE_AA)
                        else:
                            cv2.rectangle(blank_image,(v['x'],v['y'],v['width'],v['height']) ,(0,255,0), 5)
                            cv2.putText(blank_image, "%d" % (c), (v['x'],v['y']), font, 2, (255, 0, 0), 6, cv2.LINE_AA)


                        lastd[v['idv']] = (v['x']+int(v['width']/2),v['y']+int(v['height']/2))
                        newdict[v['idv']] = v['size']

                        graphit[v['idv']].append((c,v['size']))

                writer.writerow(newdict)

            fig, ax = plt.subplots()
            ax.set_title('Plant Growth')
            ax.set_xlabel('Image number', fontsize=10)
            ax.set_ylabel('Plant size', fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            for key, itm in graphit.items():
                newdatx = []
                newdaty = []
                for d in itm:
                    newdatx.append(d[0])
                    newdaty.append(d[1])

                if len(newdatx) >= 2:
                    ax.plot(newdatx, newdaty, label="Plant ID %d" % (key))


            ax.legend(loc='center left', fancybox=True, shadow=True,bbox_to_anchor=(1, 0.5))


            fig.savefig('graph-%d.pdf' % potid,bbox_inches="tight")

        cv2.imwrite('out-%d.jpg' % potid, blank_image)
           
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Interface for graphing plant growth")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=dir_path,
        help="Filename to load images from and to write output to",
        required=True
    )
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    
    args = parser.parse_args()
    generate_graph(args.in_path, args.recursive)

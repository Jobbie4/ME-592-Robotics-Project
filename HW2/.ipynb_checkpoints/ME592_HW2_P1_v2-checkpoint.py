# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 23:08:30 2025

@author: markg
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import os
import re
import torch
import cv2
from PIL import Image
from IPython.display import display

# key_id = 100
# Load the coordinates from the text file
def load_coordinates(file_path):
    coordinates = []
    # coord_log = {}
    with open(file_path, 'r') as file:
    # try:
    #     with open(file_path, "r") as file:
    #         print(f.read())
        
            for line in file:
                stripped_line = line.strip()
                # Skip comments or empty lines
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                try:
                    x, y = map(float, stripped_line.split())
                    coordinates.append((x, y))
                    # coord_log[f"X{key_id}"] = x
                    # coord_log[f"Y{key_id}"] = y
                    # key_id += 1
                except ValueError:
                    print(f"Warning: Skipping invalid data in {file_path} -> {stripped_line}")
        
    # except ValueError as e:
    #         print(f"ValueError occurred: {e}")
    # except Exception as e:
    #         print(f"Unexpected error: {e}")
    return np.array(coordinates)
    


def overlay_coordinates(image_path, neg_coords, pos_coords, rect_size=20):
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    if len(neg_coords) > 0:
        ax.scatter(neg_coords[:, 0], neg_coords[:, 1], color='red', marker='x', label='Negative Coordinates')
        for x, y in neg_coords:
            rect = patches.Rectangle((x - rect_size / 2, y - rect_size / 2), 
                                     rect_size, rect_size, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    if len(pos_coords) > 0:
        ax.scatter(pos_coords[:, 0], pos_coords[:, 1], color='blue', marker='o', label='Positive Coordinates')
        for x, y in pos_coords:
            rect = patches.Rectangle((x - rect_size / 2, y - rect_size / 2), 
                                     rect_size, rect_size, 
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

    plt.legend()
    plt.title(f'Overlay for {os.path.basename(image_path)}')
    plt.show()

def process_folder(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for png_file in png_files:
        base_name = os.path.splitext(png_file)[0]
        base_name = 'pcd0101r'
        neg_path = "C:/Mark's Python files/ME5920/HW2_GardockiFiles/01/pcd0101cneg.txt"
        pos_path = "C:/Mark's Python files/ME5920/HW2_GardockiFiles/01/pcd0101cpos.txt"
        #^^^Disable these three to test the iterative process
        
        # neg_path = next((os.path.join(folder_path, f) for f in txt_files if re.match(fr'{base_name}.*cneg/.txt', f)), None)
        # pos_path = next((os.path.join(folder_path, f) for f in txt_files if re.match(fr'{base_name}.*cpos/.txt', f)), None)
        
        # neg_path = next((os.path.join(folder_path, f) 
        #     for f in txt_files 
        #     if f.startswith(base_name) and 'cneg' in f and f.endswith('.txt')), None)

        # pos_path = next((os.path.join(folder_path, f) 
        #     for f in txt_files 
        #     if f.startswith(base_name) and 'cpos' in f and f.endswith('.txt')), None)
        
        
#WIP - still trying to figure out how I can iterate over the .txt files to find the coordinate tables for any .png
        # neg_path = next((os.path.join(folder_path, f) 
        #          for f in txt_files 
        #          if 'cneg' in f and base_name in f), None)

        # pos_path = next((os.path.join(folder_path, f) 
        #          for f in txt_files 
        #          if 'cpos' in f and base_name in f), None)



        neg_coordinates = load_coordinates(neg_path) if neg_path else np.array([])
        pos_coordinates = load_coordinates(pos_path) if pos_path else np.array([])

        if neg_coordinates.size or pos_coordinates.size:
            overlay_coordinates(os.path.join(folder_path, png_file), neg_coordinates, pos_coordinates)
        else:
            print(f'Warning: No matching coordinate files found for {png_file}')

# Folder path
folder_path = "C:/Mark's Python files/ME5920/HW2_GardockiFiles/01"
load_coordinates(folder_path)
process_folder(folder_path)


#%%
#Eric's code

img_neg = cv2.imread("C:/Users/markg/Downloads/01/pcd0102r.png",1)
img_pos = cv2.imread("C:/Users/markg/Downloads/01/pcd0102r.png",1)
img =     cv2.imread("C:/Users/markg/Downloads/01/pcd0102r.png",1)
display(Image.fromarray(img))

print(img.shape)
neg_rect_single = pd.read_csv("C:/Users/markg/Downloads/01/pcd0102cneg.txt", delimiter='\s',header=None)
neg_rect_single = neg_rect_single.to_numpy()
pos_rect_single = pd.read_csv("C:/Users/markg/Downloads/01/pcd0102cpos.txt", delimiter='\s',header=None)
pos_rect_single = pos_rect_single.to_numpy()

# This is where I'm adding in a dictionary to track the rectangle coordinates.
#   We can change the naming of the keys/ indeces later, and we can also 
#       later on append the depth data, so we would have the depth information
#       with each rectangle
i=0
RC = 0 #Rectangle Counter
pos_rectangles = {}
neg_rectangles = {}
for i in range(0,len(pos_rect_single),4):
 rectangle_pts = np.array([[pos_rect_single[i,0],pos_rect_single[i,1]], [pos_rect_single[i+1,0],pos_rect_single[i+1,1]], [pos_rect_single[i+2,0],pos_rect_single[i+2,1]], [pos_rect_single[i+3,0],pos_rect_single[i+3,1]]], np.int32)
 rectangle_pts = rectangle_pts.reshape((-1, 1, 2))
 cv2.polylines(img_pos, [rectangle_pts], isClosed=True, color=(0, 255, 0), thickness=1)
 pos_rectangles[f"PosRect_{RC}"] = rectangle_pts
 RC += 1
 
#RC = 0  #Don't necessarily need to refresh if we want this to be a unique rectangle identifier
for i in range(0,len(neg_rect_single),4):
 rectangle_pts = np.array([[neg_rect_single[i,0],neg_rect_single[i,1]], [neg_rect_single[i+1,0],neg_rect_single[i+1,1]], [neg_rect_single[i+2,0],neg_rect_single[i+2,1]], [neg_rect_single[i+3,0],neg_rect_single[i+3,1]]], np.int32)
 rectangle_pts = rectangle_pts.reshape((-1, 1, 2))
 cv2.polylines(img_neg, [rectangle_pts], isClosed=True, color=(255, 0, 0), thickness=1)
 neg_rectangles[f"NegRect_{RC}"] = rectangle_pts
 RC += 1
 
#Display the input image with rectangles overlaid
side_by_side = np.hstack((img, img_pos,img_neg))
display(Image.fromarray(side_by_side))


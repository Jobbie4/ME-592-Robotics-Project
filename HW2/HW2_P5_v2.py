# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 21:08:47 2025

@author: markg
"""

import argparse
import glob
import os

import numpy as np
from imageio import imsave

# from image import DepthImage

import tifffile


from imageio import imwrite


# path = './data/01/'
# pcds = glob.glob(os.path.join(path, 'pcd[0-9][0-9][0-9][0-9].txt'))  # Matches 'pcdXXXX.txt'
# pcds.sort()

# for pcd in pcds:
#     di, rgb_img = DepthImage.from_pcd(pcd, (480, 640))
    
#     # Save depth image
#     depth_of_name = pcd.replace('.txt', '_depth.tiff')
#     imwrite(depth_of_name, di.img.astype(np.float32))
    
#     # Save RGB image
#     rgb_of_name = pcd.replace('.txt', '_rgb.png')
#     imwrite(rgb_of_name, rgb_img)
    


file_path = "C:/Users/markg/Downloads/pcd0100_depth.tiff"
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File not found.")


try:
    with tifffile.TiffFile(file_path) as tif:
        print(tif.pages)  # Number of pages
        print("Image shape:", tif.pages[0].shape)  # Shape of first page
        print("Data type:", tif.pages[0].dtype)  # Data type
        print("Compression:", tif.pages[0].compression)  # Compression method
except Exception as e:
    print("Error:", e)
    
#%%
import cv2
import matplotlib.pyplot as plt

original = cv2.imread("C:/Users/markg/Downloads/pcd0100r.png")
rgb = cv2.imread("C:/Users/markg/Downloads/pcd0100_rgb.png")
depth = cv2.imread("C:/Users/markg/Downloads/pcd0100_depth.tiff", cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
plt.title("RGB Image")

plt.subplot(1, 3, 3)
plt.imshow(depth, cmap='gray')
plt.colorbar()
plt.title("Depth Image")

plt.show()   


#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from IPython.display import display

import tifffile
def save_rgbd_as_tiff(rgb_img, depth_img, filename):
    """
    Save RGB-D as a multi-channel TIFF file.
    """
    # Ensure depth image is loaded and has the correct shape
    if depth_img is None or depth_img.size == 0:
        raise ValueError("Depth image is empty or not loaded correctly!")

    if rgb_img.shape[:2] != depth_img.shape:
        raise ValueError(f"Shape mismatch! RGB: {rgb_img.shape}, Depth: {depth_img.shape}")

    depth_img = depth_img.astype(np.float32)  # Keep full precision

    # Ensure depth is 3D by expanding dimensions
    depth_img = np.expand_dims(depth_img, axis=-1)  # Shape: (480, 640, 1)

    # Concatenate RGB and Depth into a single image (H, W, 4)
    rgbd_img = np.concatenate((rgb_img, depth_img), axis=-1)

    # Save as TIFF
    tifffile.imwrite(filename, rgbd_img)

# Example usage
rgb = cv2.imread("C:/Users/markg/Downloads/pcd0100_rgb.png", 1) 
depth = cv2.imread("C:/Users/markg/Downloads/pcd0100_depth.tiff", cv2.IMREAD_UNCHANGED) 
save_rgbd_as_tiff(rgb, depth, "00rgbd_pcd0100.tiff")

#%%
import cv2
import numpy as np
import tifffile
import pandas as pd

# Load the 4-channel RGB-D image from TIFF
rgbd = tifffile.imread("00rgbd_pcd0100.tiff")  # Ensure this is a 4-channel TIFF
print("Loaded RGB-D shape:", rgbd.shape)  # Expected shape: (480, 640, 4)

# Extract RGB and convert to YUV
rgb = rgbd[:, :, :3]  # First 3 channels are RGB
yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)  # Convert RGB to YUV

# Extract depth (4th channel)
depth = rgbd[:, :, 3]  # Last channel is Depth

print("YUV shape:", yuv.shape)  # Should be (480, 640, 3)
print("Depth shape:", depth.shape)  # Should be (480, 640)

# Load grasping rectangles
neg_rect_single = pd.read_csv("C:/Users/markg/Downloads/pcd0100cneg.txt", delimiter='\s', header=None).to_numpy()
pos_rect_single = pd.read_csv("C:/Users/markg/Downloads/pcd0100cpos.txt", delimiter='\s', header=None).to_numpy()

# Create a copy of the depth image for visualization
depth_with_rects = depth.copy()

# Draw grasping rectangles on depth image
# for i in range(0, len(pos_rect_single), 4):
#     rectangle_pts = np.array([
#         [pos_rect_single[i, 0], pos_rect_single[i, 1]],
#         [pos_rect_single[i + 1, 0], pos_rect_single[i + 1, 1]],
#         [pos_rect_single[i + 2, 0], pos_rect_single[i + 2, 1]],
#         [pos_rect_single[i + 3, 0], pos_rect_single[i + 3, 1]]
#     ], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(depth_with_rects, [rectangle_pts], isClosed=True, color=(255, 0, 0), thickness=1)

#Repeat for negative grasping rectangles
# for i in range(0, len(neg_rect_single), 4):
#     rectangle_pts = np.array([
#         [neg_rect_single[i, 0], neg_rect_single[i, 1]],
#         [neg_rect_single[i + 1, 0], neg_rect_single[i + 1, 1]],
#         [neg_rect_single[i + 2, 0], neg_rect_single[i + 2, 1]],
#         [neg_rect_single[i + 3, 0], neg_rect_single[i + 3, 1]]
#     ], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(depth_with_rects, [rectangle_pts], isClosed=True, color=(255, 0, 0), thickness=1)

# Merge RGB and modified Depth back into a 4-channel image
rgbd_with_rects = np.dstack((rgb, depth_with_rects))  # (480, 640, 4)

# ✅ Save as a 4-channel RGB-D TIFF
tifffile.imwrite("pcd0100_rgbd_with_rects_2.tiff", rgbd_with_rects.astype(np.float32))
print("Saved RGB-D with grasping rectangles as 4-channel TIFF.")

#%%
import tifffile

# Load the saved 4-channel RGB-D image
rgbd_new = tifffile.imread("pcd0100_rgbd_with_rects_2.tiff")

print("New RGB-D Shape:", rgbd_new.shape)  # Expected output: (480, 640, 4)

# Extract RGB and Depth for visualization
rgb_new = rgbd_new[:, :, :3]
depth_new = rgbd_new[:, :, 3]

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(rgb_new.astype(np.uint8))
plt.title("RGB Image")


depth_new = np.clip(depth_new, 0, None) # there can't be negative values 



plt.subplot(1, 2, 2)
plt.imshow(depth_new, cmap='jet', vmin=0, vmax=.1) 
plt.colorbar()
plt.title("Depth Image with Grasping Rectangles")

plt.show()
    
#%% Plotting the 3d data
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the TIFF file
image_data = tiff.imread("pcd0100_rgbd_with_rects_2.tiff")

# Check shape
print(image_data.shape)


rgb = image_data[..., :3]  # RGB data
depth = image_data[..., 3]  # Depth map


h, w = depth.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))

# Normalize depth for better scaling
normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

""" 
^This normalized_depth function is where I think I might be having an issue, by setting an incorrect scalar range
for all of the point data.
 """

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(-1*x, -1*y, normalized_depth, c=rgb.reshape(-1, 3) / 255.0, s=1) #sign change for proper orientation

ax.set_title('3D Visualization from TIFF Data')
plt.show()


#%% This section plots the 3d plot in an interactive web browser window
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(z=normalized_depth, surfacecolor=rgb[..., 0])])
fig.update_traces(colorscale='gray')
fig.show()




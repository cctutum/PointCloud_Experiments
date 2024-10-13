"""
Poux, F., "How to automate LiDAR point cloud sub-sampling with Python", Towards Data Science, Nov 21, 2020. 
URL: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c

The typical sub-sampling methods for point cloud data thinning include 
the random, the minimal distance and the grid (or uniform) methods.
"""

#%%

import os
import numpy as np
import laspy as lp

#%% Read point cloud data

directory = "../data"
os.makedirs(directory, exist_ok=True)
filename = "NZ19_Wellington.las"
file_path = os.path.join(directory, filename)

pcd = lp.read(file_path)

#%% Separate points and colors

points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose() # shape: (13993118, 3)

# Optional (export data to view in CloudCompare)
# Check if the point cloud data has color info
if hasattr(pcd, 'red') and hasattr(pcd, 'green') and hasattr(pcd, 'blue'):
    colors_16bit = np.vstack((pcd.red, pcd.green, pcd.blue)).transpose()
    colors_8bit = colors_8bit = np.clip(np.round(colors_16bit / 256), 0, 255).astype(np.uint8)
    CC_data = np.hstack((points, colors_8bit))
# Save as .txt file (CloudCompare can read this file)
# Note: np.savetxt() takes a long time to write,
# np.save('output_CC.npy', CC_data) is significantly faster but it is binary format!
# or >> np.savez_compressed('output_CC.npz', data=CC_data)
save_new_CC = True
filename_CC = "output_for_CC.txt"
if filename_CC not in os.listdir(directory) and save_new_CC:
    np.savetxt(os.path.join(directory, filename_CC), 
               CC_data, delimiter=" ", fmt="%.6f") 

#%% Strategy 1: Point Cloud Random subsampling
# Select one in every 'factor' rows (points)

factor = 160
decimated_points_random = points[::factor] # [start:end:step] shape: (87457, 3)

#%% Strategy 2: Point Cloud Grid Subsampling
# The grid subsampling strategy will be based on the division of the 3D space 
# in regular voxels. For each cell of this grid, we will only keep one representative 
# point. For example, it can be the barycenter of the points in that cell, or 
# the closest point to it.

voxel_size=6
nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)



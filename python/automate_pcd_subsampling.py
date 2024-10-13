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
colors = np.vstack((pcd.red, pcd.green, pcd.blue)).transpose()

#%% Strategy 1: Point Cloud Random subsampling
# Select one in every 'factor' rows (points)

factor = 160
decimated_points_random = points[::factor] # [start:end:step] shape: (87457, 3)

#%% Strategy 2: Point Cloud Grid Subsampling
# The grid subsampling strategy will be based on the division of the 3D space 
# in regular voxels. For each cell of this grid, we will only keep one representative 
# point. For example, it can be the barycenter of the points in that cell, or 
# the closest point to it.





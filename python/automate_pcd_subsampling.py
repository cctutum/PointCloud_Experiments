"""
Poux, F., "How to automate LiDAR point cloud sub-sampling with Python", Towards Data Science, Nov 21, 2020. 
article-URL: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
data-URL = https://drive.google.com/file/d/12Iy4fkJ1i1Xh-dzGvsf_M66e8eVa1vyx/view?usp=sharing

The typical sub-sampling methods for point cloud data thinning include 
the random, the minimal distance and the grid (or uniform) methods.
"""

#%%

import os
import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
save_new_CC = False
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
# Number of voxels in each axis
XYZ_max = np.max(points, axis=0) # Upper Bound in each axis
XYZ_min = np.min(points, axis=0) # Lower Bound in each axis
nb_vox = np.ceil((XYZ_max - XYZ_min) / voxel_size) # 254 [X-axis] x 154 [Y] x 51 [Z]

# Test each voxel if it contains one or more points. If it does, we keep it.
# Below "((points - XYZ_min) // voxel_size).astype(int)" finds the voxel indices 
# in each axis for each point.
# return_inverse=True --> Returns the indices of the unique array (i.e., axis=0), 
# that can be used to reconstruct input array
non_empty_voxel_keys, inverse, nb_pts_per_unique_voxel = np.unique(
    ((points - XYZ_min) // voxel_size).astype(int), axis=0, 
    return_inverse=True, return_counts=True) 

idx_pts_vox_sorted = np.argsort(inverse)

# Next, compute the representant of each non-empty voxel: 
# Both the barycenter (grid_barycenter) and the closest point to the barycenter 
# (grid_candidate_center).
# Hint: It is recommended to use a "dictionary" to keep the points in each voxel.
# A dictionary cannot take a [i, j, k] vector of coordinates as key if it is a list, 
# but converting it to a tuple (i, j, k) will make it work!

voxel_grid = {}
grid_barycenter, grid_candidate_center = [], []
last_seen = 0
    
for idx, vox in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)] = points[ idx_pts_vox_sorted[
        last_seen : last_seen + nb_pts_per_unique_voxel[idx] ]]
    grid_barycenter.append( np.mean( voxel_grid[tuple(vox)], axis=0) )
    grid_candidate_center.append( 
        voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - 
                                              np.mean(voxel_grid[tuple(vox)], axis=0), 
                                              axis=1).argmin()]
        )
    last_seen += nb_pts_per_unique_voxel[idx]

#%% Visualize results

# Option-1 (View Random Subsampling Results)
plt.figure()
decimated_colors = colors_8bit[::factor]
ax = plt.axes(projection='3d')
xd, yd, zd = decimated_points_random[:,0], decimated_points_random[:,1], decimated_points_random[:,2]
ax.scatter(xd, yd, zd, c=decimated_colors/256, s=0.01)
plt.show()

# Option-2 (View Grid Subsampling Results - Grid barycenter)
plt.figure()
# TODO: Subsampled colors must be assigned a representative color!
ax = plt.axes(projection='3d')
grid_barycenter = np.array(grid_barycenter)
xb, yb, zb = grid_barycenter[:,0], grid_barycenter[:,1], grid_barycenter[:,2]
# ax.scatter(xb, yb, zb, c=decimated_colors/256, s=0.01)
ax.scatter(xb, yb, zb, c='blue', s=0.01)
plt.show()

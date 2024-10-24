"""
Poux, F., "3D Point Cloud Shape Detection for Indoor Modelling", Towards Data Science, Sep 7, 2023. 
article-URL: https://towardsdatascience.com/3d-point-cloud-shape-detection-for-indoor-modelling-70e36e5f2511
data-URL = https://drive.google.com/drive/folders/1sCBT1lc9A8Zn4grpxwFrBrvos86c0HZR?usp=share_link

Tutorial to automate 3D shape detection, segmentation, clustering, and voxelization 
for space occupancy 3D modeling of indoor point cloud datasets.
"""

#%%

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#%% Step-1: Read point cloud data

directory = "../data"
results_dir = "../results"
os.makedirs(directory, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
filename = "ITC_groundfloor.ply"
file_path = os.path.join(directory, filename)

pcd = o3d.io.read_point_cloud(file_path) # PointCloud with 334992 points

# To view PCD in Open3D, it is a good practice to shift our point cloud to 
# bypass the large coordinates approximation, which creates shaky visualization 
# effects
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# View pcd interactively
o3d.visualization.draw_geometries([pcd])

#%% Decimation or Downsampling: Apply random sampling method that can effectively 
# reduce point cloud size
retained_ratio = 0.2
sampled_pcd = pcd.random_down_sample(retained_ratio)

# View downsampled pcd
o3d.visualization.draw_geometries([sampled_pcd], window_name = "Random Sampling")

"""
Note of random downsampling:
When studying 3D point clouds, random sampling has limitations that could result 
in missing important information and inaccurate analysis. It doesn’t consider 
the spatial component or relationships between the points. Therefore, it’s 
essential to use other methods to ensure a more comprehensive analysis.
"""

#%% Statistical outlier removal
# Remove points that are further away from their neighbors compared to the 
# average for the point cloud.
nn = 16 
# number of neighbors to consider for calculating the average distance for a given point
std_multiplier = 10 
# threshold level based on the std. deviation of the average distances across 
# the point cloud. The lower this number, the more aggressive the filter will be.
filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)

# View the outliers (in red)
outliers = pcd.select_by_index(filtered_idx, invert= True)
outliers.paint_uniform_color([1, 0, 0]) # red
o3d.visualization.draw_geometries([filtered_pcd, outliers])

#%% Apply voxel (grid)-based sampling technique to downsample the data further
voxel_size = 0.05
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)

# View final downsamped pcd
o3d.visualization.draw_geometries([pcd_downsampled])

#%% Extract PCD Normals
# A point cloud normal refers to the direction of a surface at a specific point 
# in a 3D point cloud. It can be used for segmentation by dividing the point 
# cloud into regions with similar normals, for example. 
# In this case, normals will help identify objects and surfaces within the point 
# cloud, making it easier to visualize.

# First, define the average distance between each point in the point cloud and 
# its neighbors:
nn_distance = 0.05

# Next, we use this information to extract a limited 'max_nn' points within a 
# radius 'radius_normals' to compute a normal for each point in the 3D point cloud:
radius_normals = nn_distance * 4
pcd_downsampled.estimate_normals(search_param=
                                 o3d.geometry.KDTreeSearchParamHybrid(
                                     radius= radius_normals, 
                                     max_nn= 16), 
                                 fast_normal_computation= True)

# View he downsampled point cloud that have estimated normals
pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled, outliers])

#%% RANSAC Parameter Setting


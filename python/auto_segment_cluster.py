"""
Poux, F., "How to automate 3D point cloud segmentation and clustering with Python", Towards Data Science, May 12, 2021. 
article-URL: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
data-URL = https://drive.google.com/file/d/1CJrH9eqzOte3PHJ_g8iLifJFE_cBGg6b/view?usp=sharing

The typical sub-sampling methods for point cloud data thinning include 
the random, the minimal distance and the grid (or uniform) methods.
"""

#%%

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%% Read point cloud data

directory = "../data"
os.makedirs(directory, exist_ok=True)
filename = "TLS_kitchen.ply"
file_path = os.path.join(directory, filename)

pcd = o3d.io.read_point_cloud(file_path) # PointCloud with 511026 points

# Estimate normals (Optional)
#TODO: # generalize parameters below (radius, max_nn)
kdtree_search = o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1, max_nn= 16) 
pcd.estimate_normals(search_param= kdtree_search, 
                     fast_normal_computation= True)

#%% RANSAC segmentation for planar shape detection

plane_model, inliers = pcd.segment_plane(distance_threshold= 0.01, # automate this!
                                         ransac_n= 3, # 3 for plane
                                         num_iterations= 1000)

inlier_pcd = pcd.select_by_index(inliers)
outlier_pcd = pcd.select_by_index(inliers, invert=True)

inlier_pcd.paint_uniform_color([1, 0, 0]) # Red
outlier_pcd.paint_uniform_color([0.6, 0.6, 0.6]) # Grey

o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])

#%% DBSCAN cllustering on a smaller sample

# DBSCAN params: radius of 5 cm, min. 10 points to form a cluster
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10)) 

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

#%%






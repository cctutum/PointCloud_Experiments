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

#%% Step-1: Read point cloud data

directory = "../data"
os.makedirs(directory, exist_ok=True)
filename = "TLS_kitchen.ply"
file_path = os.path.join(directory, filename)

pcd = o3d.io.read_point_cloud(file_path) # PointCloud with 511026 points

# Estimate normals (Optional)
#TODO: # generalize parameters below   (radius, max_nn)
kdtree_search = o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1, max_nn= 16) 
pcd.estimate_normals(search_param= kdtree_search, 
                     fast_normal_computation= True)

#%% Step-2: RANSAC segmentation for planar shape detection

plane_model, inliers = pcd.segment_plane(distance_threshold= 0.01, # automate this!
                                         ransac_n= 3, # 3 for plane
                                         num_iterations= 1000)

inlier_pcd = pcd.select_by_index(inliers)
outlier_pcd = pcd.select_by_index(inliers, invert=True)

inlier_pcd.paint_uniform_color([1, 0, 0]) # Red
outlier_pcd.paint_uniform_color([0.6, 0.6, 0.6]) # Grey

o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])

#%% Step-3: DBSCAN cllustering on a smaller sample

# Let's select a sample, where we assume we got rid of all the planar regions
pcd_sample = o3d.io.read_point_cloud(os.path.join(directory, "TLS_kitchen_sample.ply"))
kdtree_search = o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1, max_nn= 16) # 10 cm
pcd_sample.estimate_normals(search_param= kdtree_search, 
                            fast_normal_computation= True)

# Skip RANSAC-step!

# DBSCAN params: radius of 5 cm, min. 10 points to form a cluster
d_threshold = 0.05
labels = np.array(pcd_sample.cluster_dbscan(eps= d_threshold, min_points= 10)) 
# The labels vary between -1 and n, where -1 indicate it is a “noise” point and 
# values 0 to n are then the cluster labels given to the corresponding point.

# Color the point cloud
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1)) # [R. G, B, Opacity]
colors[labels < 0] = 0
colors_o3d = o3d.utility.Vector3dVector(colors[:, :3]) # Remove opacity
pcd_sample.colors = colors_o3d
o3d.visualization.draw_geometries([pcd_sample])

#%% Step 4: Scaling and automation

segment_models = {}
segments = {}

max_plane_idx = 20

rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    # RANSAC
    segment_models[i], inliers = rest.segment_plane(distance_threshold= 0.01,
                                                    ransac_n= 3,
                                                    num_iterations= 1000)
    segments[i] = rest.select_by_index(inliers) # temporary
    # DBSCAN
    labels = np.array(segments[i].cluster_dbscan(eps= d_threshold*10, 
                                                 min_points= 10))
    candidates = [ len(np.where(labels==j)[0]) for j in np.unique(labels) ]
    best_candidate = int( np.unique(labels)
                         [np.where(candidates == np.max(candidates))[0]] )
    rest = rest.select_by_index(inliers, invert=True)
    rest += segments[i].select_by_index(
        list(np.where(labels != best_candidate)[0]) )
    segments[i] = segments[i].select_by_index(
        list(np.where(labels == best_candidate)[0]) )
    segments[i].paint_uniform_color( list(colors[:3]) )
    print(f"plane: {i} / {max_plane_idx} done.")
    
# Clustering the remaining points with DBSCAN
labels = np.array(rest.cluster_dbscan(eps= d_threshold, min_points= 5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + 
                                  [rest] )



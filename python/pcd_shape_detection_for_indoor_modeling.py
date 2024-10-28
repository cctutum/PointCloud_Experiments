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

# If we were to select distance_threshold automatically (however we will not 
# continue with this):
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance()) 
# This gives ~5 cm as we set the voxel_size = 0.05 before

#%% Point Cloud Segmentation with RANSAC

distance_threshold = 0.1
ransac_n = 3
num_iterations = 1_000

plane_model, inliers = pcd.segment_plane(
    distance_threshold= distance_threshold,
    ransac_n= 3, # for plane
    num_iterations= 1000)
[a, b, c, d] = plane_model

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert= True)

#Paint the clouds
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

#Visualize the inliers and outliers
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#%% Multi-Order RANSAC & Euclidean Clustering (DBSCAN)

segment_models = {} # a, b, c, d
segments = {} # inlier points for each segment

max_plane_idx = 10

# RANSAC parameters
d_threshold = 0.1
ransac_n = 3
num_iterations = 1_000

# DBSCAN parameters
epsilon = 0.15
min_cluster_points = 5

rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    # RANSAC
    segment_models[i], inliers = rest.segment_plane(
                                        distance_threshold= d_threshold,
                                        ransac_n= ransac_n,
                                        num_iterations= num_iterations)
    segments[i] = rest.select_by_index(inliers)
    # DBSCAN
    labels = np.array( segments[i].cluster_dbscan(
                                            eps= epsilon, 
                                            min_points= min_cluster_points))
    candidates = [ len(np.where(labels == j)[0]) for j in np.unique(labels) ]
    max_index = np.argmax(candidates)
    best_candidate = int(np.unique(labels)[max_index])
    rest = rest.select_by_index(inliers, invert= True)
    rest += segments[i].select_by_index(
                                list(np.where(labels != best_candidate)[0]) )
    # Segmented plane (best candidate) updated after clustering
    segments[i] = segments[i].select_by_index(
        list(np.where(labels == best_candidate)[0]) )
    segments[i].paint_uniform_color( list(colors[:3]) )
    print(f"Segment= {i}/{max_plane_idx} done.")
    
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + 
                                  [rest])
    
#%% Euclidean Clustering Refinement (on the remaining points)
    
# You can start with viewing the remaining points
# o3d.visualization.draw_geometries([rest])
    
labels = np.array( rest.cluster_dbscan(eps= 0.15, min_points= 10) )

# Available colormaps
# from matplotlib import colormaps
# print(list(colormaps)) # list of 170 colormaps

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1)) # [R. G, B, Opacity]
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3]) # Remove opacity
o3d.visualization.draw_geometries([rest]) # these points are mostly assigned black color

# OPTIONAL: To view only points that have colors other than black 
# non_zero_mask = np.any(colors != 0, axis= 1)
# non_zero_indices = np.where(non_zero_mask)[0]
# rest_filtered = rest.select_by_index(non_zero_indices)
# o3d.visualization.draw_geometries([rest_filtered])
    
# OPTIONAL: View all the segments and rest point clouds with final color assignment 
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + 
                                  [rest])

#%% Voxelization and Labelling
# By dividing a point cloud into small cubes, it becomes easier to understand 
# a model's occupied and empty spaces.

voxel_size = 0.5

min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()    
    

'''
Let's illustrate the case where you want to have voxels of "structural" 
elements vs voxels of clutter that do not belong to structural elements.
Without labeling, we could guide our choice based on whether or not they 
belong to RANSAC segments or other segments; This means first concatenating 
the segments from the RANSAC pass:
'''
pcd_ransac = o3d.geometry.PointCloud()
for i in segments:
    pcd_ransac += segments[i]

# Voxel-ize structural elements
voxel_grid_structural = o3d.geometry.VoxelGrid.\
                                create_from_point_cloud(pcd_ransac, 
                                                        voxel_size= voxel_size)
                                
# Voxel-ize clutter (rest) elements
rest.paint_uniform_color([0.1, 0.1, 0.8])
voxel_grid_clutter = o3d.geometry.VoxelGrid.\
                                create_from_point_cloud(rest, 
                                                        voxel_size= voxel_size)
                                
# View the voxels
o3d.visualization.draw_geometries([voxel_grid_structural])
# o3d.visualization.draw_geometries([voxel_grid_clutter, voxel_grid_structural])

#%% Spatial Modeling

'''
Define a function that fits a voxel grid and returns both filled and empty spaces:
(1) Determine the minimum and maximum coordinates of the point cloud, 
(2) Calculate the dimensions of the voxel grid, 
(3) Create an empty voxel grid, 
(4) Calculate the indices of the occupied voxels, and 
(5) Mark occupied voxels as True
'''
def fit_voxel_grid(point_cloud, voxel_size, min_b= False, max_b= False):
    # (1) Determine the minimum and maximum coordinates of the point cloud
    if type(min_b) == bool or type(max_b) == bool:
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
    else:
        min_coords = min_b
        max_coords = max_b
    # (2) Calculate the dimensions of the voxel grid
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    # (3) Create an empty voxel grid
    voxel_grid = np.zeros(grid_dims, dtype=bool)
    # (4) Calculate the indices of the occupied voxels
    indices = ((point_cloud - min_coords) / voxel_size).astype(int)
    # (5) Mark occupied voxels as True
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return voxel_grid, indices


voxel_size = 0.3

# Get the bounds of the original point cloud
min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

ransac_voxels, idx_ransac = fit_voxel_grid(pcd_ransac.points, voxel_size, min_bound, max_bound)
rest_voxels, idx_rest = fit_voxel_grid(rest.points, voxel_size, min_bound, max_bound)

# Gather the filled voxels from RANSAC Segmentation
filled_ransac = np.transpose(np.nonzero(ransac_voxels))

# Gather the filled remaining voxels (not belonging to any segments)
filled_rest = np.transpose(np.nonzero(rest_voxels))

# Compute and gather the remaining empty voxels
total = pcd_ransac + rest
total_voxels, idx_total = fit_voxel_grid(total.points, voxel_size, min_bound, max_bound)
empty_indices = np.transpose(np.nonzero(~total_voxels))


#%% Exporting 3D Datasets



"""
Poux, F., "5-Step Guide to generate 3D meshes from point clouds with Python", Towards Data Science, Apr 21, 2020. 
article-URL: https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
data-URL = https://drive.google.com/file/d/12Iy4fkJ1i1Xh-dzGvsf_M66e8eVa1vyx/view?usp=sharing

Tutorial to generate 3D meshes (.obj, .ply, .stl, .gltf) automatically from 
3D point clouds using python.
"""

#%%

import os
import numpy as np
import open3d as o3d

#%% Step-1: Read point cloud data

directory = "../data"
results_dir = "../results"
os.makedirs(directory, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
filename = "sample_w_normals.xyz"
file_path = os.path.join(directory, filename)

point_cloud = np.loadtxt(file_path, skiprows= 1) 
# point_cloud.shape = (1505010, 9)

xyz = point_cloud[:, :3] # (1_505_010, 3)
rgb = point_cloud[:, 3:6] / 255
normals = point_cloud[:, 6:]

# Transform the point_cloud variable type from Numpy to 
# Open3D o3d.geometry.PointCloud type for further processing
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
pcd.normals = o3d.utility.Vector3dVector(normals)

o3d.visualization.draw_geometries([pcd])

#%% Choose a meshing strategy

# Strategy-1: Ball-Pivoting Algorithm (BPA)

# In theory, the "diameter of the ball" should be slightly larger than the 
# average distance between points.

distances = pcd.compute_nearest_neighbor_distance() # I wonder if this uses KDTree?
# distances: List[float]
# distances.shape = xyz.shape[0] = 1_505_010
avg_dist = np.mean(distances)
radius = 3 * avg_dist






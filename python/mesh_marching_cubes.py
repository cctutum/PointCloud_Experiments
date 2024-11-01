"""
Poux, F., "3D Mesh fom Point CLoud: Python with Marching Cubes Tutorial", 3D Data Academy, October, 2024. 
article-URL: https://learngeodata.eu/3d-mesh-from-point-cloud-python-with-marching-cubes-tutorial/?utm_source=mailpoet&utm_medium=email&utm_source_platform=mailpoet&utm_campaign=3D%20Youtube

This tutorial dives deep into the Marching Cubes algorithm, a powerful technique 
for meshing 3D point clouds using Python. Given point cloud is transformed into 
a 3D mesh, various parameters are experimented, and a simple web app is built 
with a graphical user interface (GUI). This method bypasses the limitations of 
other reconstruction techniques like Poisson reconstruction, ball pivoting, and 
Delaunay triangulation.
"""

#%%

import os
import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree
import gradio as gr

#%% Step-1: Read point cloud data

directory = "../data"
results_dir = "../results"
os.makedirs(directory, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
filename = "holidays-hat.xyz"
file_path = os.path.join(directory, filename)

pcd = np.loadtxt(file_path, skiprows= 1, delimiter= ";")

xyz = pcd[:, :3]
rgb = pcd[:, 3:]

#%% Step-2: Bounds Computation

mins = np.min(xyz, axis=0)
maxs = np.max(xyz, axis=0)

#%% Step-3: Creating a 3D Grid for the 3D Point Cloud

voxel_size = 0.1 
iso_level_percentile = 20

x = np.arange(mins[0], maxs[0], voxel_size)
y = np.arange(mins[1], maxs[1], voxel_size)
z = np.arange(mins[2], maxs[2], voxel_size)
x, y, z = np.meshgrid(x, y, z, indexing='ij')

#%% Step-4: KD-Tree for Efficient Nearest Neighbor Search

tree = cKDTree(xyz)

#%% Step-5: Scalar Field: Distance to Nearest Point

"""
For each grid point (corner of a voxel), we are going to calculate the distance 
to the nearest point in the point cloud using the KD-Tree. This distance value 
becomes the scalar field value at that grid point.
"""








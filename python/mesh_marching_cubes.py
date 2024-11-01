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
filename = "holidays-hat-subsampled.xyz"
# "holidays-hat.xyz" has 1_000_000 points. KDTree computation and nearest neighbor 
# search were very time consuming, so I had to subsample the point cloud to ~15_000
# points in CLoudCOmpare by Space Sampling (distance between points was set to 0.2)
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

grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
distances, _ = tree.query(grid_points)
scalar_field = distances.reshape(x.shape)

#%% Step-6: Determine the Iso Level Based on Percentile of Distances
iso_level = np.percentile(distances, iso_level_percentile)

#%% Step-7: Apply Marching Cubes

verts, faces, _, _ = measure.marching_cubes(scalar_field, level= iso_level)

#%% Step-8: Scale and Translate Vertices Back to Original Coordinate System

verts = verts * voxel_size + mins

#%% Step-9: Creating 3D Mesh

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)

#%% Step-10: Computing 3D Mesh (vertex) Normals

mesh.compute_vertex_normals()

#%% Step-11: 3D Mesh Visualization

o3d.visualization.draw_geometries([mesh], mesh_show_back_face= True)











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

pcd = np.loadtxt(file_path, skiprows= 1)
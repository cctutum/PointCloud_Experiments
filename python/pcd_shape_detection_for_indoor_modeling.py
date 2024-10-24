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
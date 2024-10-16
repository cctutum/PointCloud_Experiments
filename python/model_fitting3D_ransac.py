"""
Poux, F., "3D Model Fitting for Point Clouds with RANSAC and Python", Towards Data Science, Oct 3, 2022. 
article-URL: https://towardsdatascience.com/3d-model-fitting-for-point-clouds-with-ransac-and-python-2ab87d5fd363
data-URLs = 
The Researcher Desk: https://drive.google.com/file/d/1OBgYx5c0-m4DGY1Lz2nl4NR56atvwU1e/view?usp=sharing
The Car: https://drive.google.com/file/d/12dAMGQmET2NIdRCXQxWnhLDf2mZkngEC/view?usp=sharing
The Playground: https://drive.google.com/file/d/1rakvffprfchT_KmEUNw35GLY5OKC-67W/view?usp=sharing


3D Point Cloud binary segmentation: RANSAC implementation from scratch.

The general form of the equation of a plane in R^3 is ax + by + cz + d = 0.
The a, b, and c constants are the components of the normal vector n=(a, b, c), 
which is perpendicular to the plane or any vector parallel to the plane. 
The d constant will shift the plane from the origin.

A point p = (x, y, z) belongs to the plane guided by the normal vector n, 
if it satisfies the equation.

"""

#%%

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.express as px
from sklearn.neighbors import KDTree

#%% Step-1: Read point cloud data

directory = "../data"
os.makedirs(directory, exist_ok=True)
filename = "the_researcher_desk.xyz"
file_path = os.path.join(directory, filename)

pcd = np.loadtxt(file_path, skiprows= 1)

xyz = pcd[:, :3]
rgb = pcd[:, 3:]

#%% Set parameters

# Interactive 2D-plot to have an idea about the distances and measurement system.
# fig = px.scatter(x=xyz[:,0], y=xyz[:,1], color=xyz[:,2])
# fig.write_html("the_researcher_desk.html") # fig.show() does not work in Spyder IDE!

# Setting up the distance (d_threshold) separates inliers from outliers of a plane:
# Average distance between neighboring points can be computed for the threshold
# (below), 
# Manual way: however we set it up manually here, i.e., 5 mm * 10 (0.005 * 10 = 0.05 m)
d_threshold = 0.05
iterations = 1000

# Automatic way:
tree = KDTree(np.array(xyz), leaf_size=2)  









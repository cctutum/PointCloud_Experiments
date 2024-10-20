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
# Manual way (however we set it up manually here, 
# i.e., 5 mm * 10 (0.005 * 10 = 0.05 m):
d_threshold = 0.05
iterations = 1000

# Automatic way (to speed up the process of querying the nearest neighbors 
# for each point):
tree = KDTree(xyz, leaf_size= 2)  

# We can query the k-nearest neighbors for each point in the point cloud 
nearest_dist, nearest_ind = tree.query(xyz, k= 8) 
# Output: tuple of two numpy arrays
# nearest_dist.shape = number of points x k -> 
#           distances (sorted in ascending order) of k-neighbors for each point
# We can ignore the first column, which is always zeros, because the every point
# is compared to itself. So, the first column shows the distance to the closest
# point. 
# nearest_ind.shape = number of points x k -> indices of points

# Average distance
mean_dist = np.mean(nearest_dist[:, 1:], axis= 0)
# array([0.0046, 0.0052 , 0.0059, 0.0067, 0.0074, 0.0081, 0.0087]) 
# Average distance to nearest neighbor = 4.6 mm

# To get a local representation of the mean distance of each point to its 
# nth closest neighbors
mean_dist_nthClosest = np.mean(nearest_dist[:, 1:]) # 6.7 mm

# It is recommended to query 8 to 15 neighbors and to average it for a good
# local representation of the noise ratio in the point cloud.



 








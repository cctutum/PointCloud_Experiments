"""
Poux, F., "5-Step Guide to generate 3D meshes from point clouds with Python", Towards Data Science, Apr 21, 2020. 
article-URL: https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
data-URL = https://drive.google.com/file/d/12Iy4fkJ1i1Xh-dzGvsf_M66e8eVa1vyx/view?usp=sharing

Tutorial to generate 3D meshes (.obj, .ply, .stl, .gltf) automatically from 
3D point clouds using python.
"""

#%%

import numpy as np
import open3d as o3d

#%%
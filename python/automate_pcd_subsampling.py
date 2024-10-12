"""
Poux, F., "How to automate LiDAR point cloud sub-sampling with Python", Towards Data Science, Nov 21, 2020. 
URL: https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c

The typical sub-sampling methods for point cloud data thinning include 
the random, the minimal distance and the grid (or uniform) methods.
"""

#%%

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%%
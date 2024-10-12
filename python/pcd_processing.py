"""
Reference:
Poux, F., "Discover 3D Point Cloud Processing with Python", Towards Data Science, Apr 13, 2020. 
URL: https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7
"""

#%%

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# to switch to 'automatic' (i.e. interactive) plots.
# %matplotlib auto # it works in Spyder IDE without a problem, but the editor 
# shows "invalid syntax" error  
# You can also set it on by Tools > Preferences > IPython Console > Graphics 
# > Graphics backend > Automatic/Inline

#%%

# Ensure the 'data' directory exists
directory = "../data"
os.makedirs(directory, exist_ok=True)
filename = "sample.xyz"
file_path = os.path.join(directory, filename)

# NOTE:
# Manually download "sample.xyz" from the publicly shared Google-Drive-Folder:
# https://drive.google.com/drive/folders/1Ih_Zz9a6UcbUlaA-puEB_is7DYvXrb4w
# Automation of this task requires the use of Google Drive API, because parsing
# the folder's HTML directly using "beautifulsoup4" module does not work.

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file '{filename}' does not exist in the directory '{directory}'")
else:
    print(f"The file '{filename}' was found in the directory '{directory}'")    
 
pcd = np.loadtxt(file_path, skiprows=1, max_rows=1_000_000) # X Y Z R G B
# original row size = 1_505_010

#%% Extracting desired attributes

xyz = pcd[:, :3]
rgb = pcd[:, 3:]

#%% Attribute-based data analysis

zMean = xyz[:, 2].mean() # or np.mean(xyz, axis=0)[2]
print(zMean, '\n')

pcd_1meter_within_zMean = pcd[ np.abs(pcd[:,2] - zMean) <= 1. ]
print(pcd_1meter_within_zMean, '\n')
print(pcd_1meter_within_zMean.shape)

#%% Basic 3D Visualisation

xyz = pcd_1meter_within_zMean[:, :3]
rgb = pcd_1meter_within_zMean[:, 3:]

ax = plt.axes(projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb/255, s=0.01) # s: Marker size
plt.show()
# You can turn on the interactive mode without using the magic command %matplotlib auto
plt.ioff() # plt.ion()


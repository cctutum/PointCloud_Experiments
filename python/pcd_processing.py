"""
Reference:
Poux, F., "Discover 3D Point Cloud Processing with Python", Towards Data Science, Apr 13, 2020. 
URL: https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7
"""

#%%

import numpy as np
import os

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

xyz = pcd[:, :3]
rgb = pcd[:, 3:]

#%%


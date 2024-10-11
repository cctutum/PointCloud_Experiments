"""
Reference:
Poux, F., "Discover 3D Point Cloud Processing with Python", Towards Data Science, Apr 13, 2020. 
URL: https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7
"""

#%%

import numpy as np
import os
import requests

#%%

# Ensure the 'data' directory exists
data_path = "../data"
os.makedirs(data_path, exist_ok=True)

url = "https://drive.google.com/drive/folders/1Ih_Zz9a6UcbUlaA-puEB_is7DYvXrb4w?usp=sharing"
filename = "sample.xyz"

if filename not in os.listdir(data_path):
    response = requests.get(f"{url}/{filename}")
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in write binary mode
        with open(f"data/{filename}", 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"File downloaded successfully: data/{filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
    
#%%
    
file_path = "../data/sample.xyz"
pcd = np.loadtxt(file_path, skiprows=1, max_rows=1_000_000) # X Y Z R G B
# original row size = 1_505_010

xyz = pcd[:, :3]
rgb = pcd[:, 3:]

#%%


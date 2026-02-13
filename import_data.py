import kagglehub
import shutil
import os

cache_path = kagglehub.dataset_download("saumyapatel/traffic-vehicles-object-detection")

destination = "./traffic-vehicles-dataset"

if not os.path.exists(destination):
    shutil.move(cache_path, destination)
    print(f"Dataset moved to: {os.path.abspath(destination)}")
else:
    print(f"Folder '{destination}' already exists.")
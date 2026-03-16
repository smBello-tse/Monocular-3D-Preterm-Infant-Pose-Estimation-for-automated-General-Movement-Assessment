import csv
import numpy as np
import os
from lib.utils.agma import *

og_dir = "data/raw_data"
subset_list = ["train", "test"]#os.listdir(og_dir)

for subset in subset_list:
    print("\n\n########################################################################")
    print(subset)
    patients_dir = os.path.join(og_dir, subset)
    patients_list = os.listdir(patients_dir)
    for patient in patients_list:
        print(patient)
        # Load 3d pose in camera coordinates (centimeters)
        files_path = os.path.join(patients_dir, patient)
        files = os.listdir(files_path)
        file3d = [os.path.join(files_path, f) for f in files if f.endswith("3D_coords_filtered.csv")]
        file2d = [os.path.join(files_path, f) for f in files if f.endswith("left_data_filtered.csv")]
        input = read_GT2D_csv(file2d[0], True)
        joint_3d_cam = read_GT3D_csv(file3d[0]) * 10.
        np.save(os.path.join(files_path, "2d_pose.npy"), input) 
        np.save(os.path.join(files_path, "3d_pose.npy"), joint_3d_cam)
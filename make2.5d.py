import csv
import numpy as np
import os
from lib.utils.agma import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="train", help="Whether train, test or infer")
    opts = parser.parse_args()
    return opts
if __name__ == "__main__":
    opts = parse_args()
    og_dir = "data/raw_data"
    subset_list = ["train", "test"]#os.listdir(og_dir)
    camera = {'fx': 532.8180541992188,
            'fy': 532.8180541992188,
            'cx': 639.40,
            'cy': 374.1215}
    root_idx = 5 #left shoulder in COCO format
    rect_size = 1000 #in millimeters
    subset = opts.subset

    print("\n\n########################################################################")
    print(subset)
    patients_dir = os.path.join(og_dir, subset)
    patients_list = os.listdir(patients_dir)
    for patient in patients_list:
        print(patient)
        files_path = os.path.join(patients_dir, patient)
        input = np.load(os.path.join(files_path, "2d_pose.npy"))
        joint_3d_cam = np.load(os.path.join(files_path, "3d_pose.npy"))

        # Compute 3d_image and 2.5d_image, with depth in millimeters
        joint_3d_image, ratios = camera_to_image_frame(joint_3d_cam, camera, root_idx, rect_size)
        joint_25d_image = np.zeros((joint_3d_image.shape[0], joint_3d_image.shape[1],
                                    4))  # The fourth coordinates is the ratio. It is constant within a frame.
        for t in range(joint_3d_image.shape[0]):
            for j in range(joint_3d_image.shape[1]):
                joint_25d_image[t, j, 0] = joint_3d_image[t, j, 0] / ratios[t]
                joint_25d_image[t, j, 1] = joint_3d_image[t, j, 1] / ratios[t]
                joint_25d_image[t, j, 2] = joint_3d_image[t, j, 2] / ratios[t]
                joint_25d_image[t, j, 3] = ratios[t]

        # saving
        path_3d_image = os.path.join(files_path, "3d_image_pose.npy")
        path_25d_image = os.path.join(files_path, "25d_pose.npy")
        ratios_path = os.path.join(files_path, "ratios.npy")
        if opts.subset == "infer": np.save(ratios_path, joint_25d_image[:, :, 3])
        else:
            np.save(path_3d_image, joint_3d_image)
            np.save(path_25d_image, joint_25d_image)




import csv
import numpy as np
import os

from lib.data.dataset_wild_AGMA_2 import coco2h36m
from lib.utils.agma import *
import cv2
import pickle

nb_frames = 1800
nb_joints = 17
nb_2d_features = 3

view = "r" #"" for left and "r" for right
#train
patients_dir = f"data/raw_data/train"
patients_list = os.listdir(patients_dir)
i = 0
joint_2d = np.zeros((nb_frames * len(patients_list), nb_joints, nb_2d_features))
ref_joint_2d = np.zeros((nb_frames * len(patients_list), nb_joints, nb_2d_features))
joint3d_image = np.zeros((nb_frames * len(patients_list), nb_joints, 3))
joint3d_cam = np.zeros((nb_frames * len(patients_list), nb_joints, 3))
joint25d_image = np.zeros((nb_frames * len(patients_list), nb_joints, 3))
camera_name = []
source = []
factors = np.zeros((nb_frames * len(patients_list),))
res_w, res_h = 880, 720
os.makedirs("data/motion3d", exist_ok=True)


for patient in patients_list:
    #print(patient)
    files_path = os.path.join(patients_dir, patient)
    files = os.listdir(files_path)
    #cap = cv2.VideoCapture(os.path.join(files_path, [f for f in files if f.endswith(".mp4")][0]))
    #res_w, res_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera_name += [str(res_w) + str(res_h) for i in range(nb_frames)]

    joint_2d[i * nb_frames:(i+1)*nb_frames, :, :] = np.load(os.path.join(files_path, [f for f in files if f.endswith("2d_pose.npy")][0]))
    ref_joint_2d[i * nb_frames:(i + 1) * nb_frames, :, :] = np.load(
        os.path.join(files_path, [f for f in files if f.endswith("2d_pose.npy")][0]))
    joint3d_image[i * nb_frames:(i+1)*nb_frames, :, :] = np.load(os.path.join(files_path, [f for f in files if f.endswith("3d_image_pose.npy")][0]))
    joint3d_cam[i * nb_frames:(i + 1) * nb_frames, :, :] = np.load(
        os.path.join(files_path, [f for f in files if f.endswith("3d_pose.npy")][0]))
    joint25d_image_total = np.load(os.path.join(files_path, [f for f in files if f.endswith("25d_pose.npy")][
        0]))  # Dimension nb_framesx17x4 x, y, z et facteur inverse
    joint25d_image[i * nb_frames:(i + 1) * nb_frames, :, :] = joint25d_image_total[:, :, :3]
    factors[i * nb_frames:(i + 1) * nb_frames] = np.power(joint25d_image_total[:, 0, 3], -1)
    source += [patient for i in range(nb_frames)]
    i = i + 1
    print(f"Patient: {patient} res_w: {res_w} res_h: {res_h} camera_name: {camera_name[-1]}")
joint_2d = coco2h36m(joint_2d)
ref_joint_2d = coco2h36m(ref_joint_2d)
joint3d_image = coco2h36m(joint3d_image)
joint3d_cam = coco2h36m(joint3d_cam)
joint_2d = joint_2d[:, :, :2]
camera_name = np.array(camera_name)
print(joint3d_cam.shape)
train_dt = {"joint_2d": joint_2d,
            "joint3d_image": joint3d_image,
            "camera_name": camera_name,
            "source": source,
            "2.5d_factor": factors,
            "joint3d_cam": joint3d_cam,
            "ref_joint_2d": ref_joint_2d}


print("Train done!")






#####test
patients_dir = f"data/raw_data/test"
patients_list = os.listdir(patients_dir)
i = 0
joint_2d = np.zeros((nb_frames * len(patients_list), 17, 3))
ref_joint_2d = np.zeros((nb_frames * len(patients_list), 17, 3))
joint3d_image = np.zeros((nb_frames * len(patients_list), 17, 3))
joint25d_image = np.zeros((nb_frames * len(patients_list), 17, 3))
factors = np.zeros((nb_frames * len(patients_list), ))
joint3d_cam = np.zeros((nb_frames * len(patients_list), 17, 3))
camera_name = []
source = []
action = []



for patient in patients_list:
    files_path = os.path.join(patients_dir, patient)
    files = os.listdir(files_path)
    #cap = cv2.VideoCapture(os.path.join(files_path, [f for f in files if f.endswith(".mp4")][0]))
    #res_w, res_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera_name += [str(res_w) + str(res_h) for i in range(nb_frames)]
    ref_joint_2d[i * nb_frames:(i + 1) * nb_frames, :, :] = np.load(
        os.path.join(files_path, [f for f in files if f.endswith("2d_pose.npy")][0]))
    joint3d_cam[i * nb_frames:(i + 1) * nb_frames, :, :] = np.load(
        os.path.join(files_path, [f for f in files if f.endswith("3d_pose.npy")][0]))
    joint_2d[i * nb_frames:(i+1)*nb_frames, :, :] = np.load(os.path.join(files_path, [f for f in files if f.endswith( "2d_pose.npy")][0]))
    joint3d_image[i * nb_frames:(i+1)*nb_frames, :, :] = np.load(os.path.join(files_path, [f for f in files if f.endswith("3d_image_pose.npy")][0]))
    joint25d_image_total = np.load(os.path.join(files_path, [f for f in files if f.endswith("25d_pose.npy")][0])) #Dimension nb_framesx17x4 x, y, z et facteur inverse
    joint25d_image[i * nb_frames:(i + 1) * nb_frames, :, :] = joint25d_image_total[:, :, :3]
    factors[i * nb_frames:(i + 1) * nb_frames] = np.power(joint25d_image_total[:, 0, 3], -1)
    source += [patient for i in range(nb_frames)]
    action += ["Baby" for i in range(nb_frames)]
    i = i + 1
    print(f"Patient: {patient} res_w: {res_w} res_h: {res_h} camera_name: {camera_name[-1]}")

joint_2d = coco2h36m(joint_2d)
ref_joint_2d = coco2h36m(ref_joint_2d)
joint3d_image = coco2h36m(joint3d_image)
joint25d_image = coco2h36m(joint25d_image)
joint3d_cam = coco2h36m(joint3d_cam)
joint_2d = joint_2d[:, :, :2]
camera_name = np.array(camera_name)
test_dt = {"joint_2d": joint_2d,
            "joint3d_image": joint3d_image,
            "camera_name": camera_name,
            "source": source,
           "action": action,
           "2.5d_factor": factors,
           "joints_2.5d_image": joint25d_image,
           "joint3d_cam": joint3d_cam,
           "ref_joint_2d": ref_joint_2d}
print("Test done!")


dt = {"train": train_dt,
      "test": test_dt}

with open(f"data/motion3d/data.pkl", "wb") as f:
    pickle.dump(dt, f)

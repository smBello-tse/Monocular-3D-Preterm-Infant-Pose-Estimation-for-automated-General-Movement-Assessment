# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:27:56 2025

@author: soule
"""

import csv
import numpy as np
import ast  # To safely convert string representations of lists
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel, xlabel
import random
import torch


def read_GT3D_csv(file_path):

    """
    Reads a CSV file and extracts joint coordinates into a NumPy array of shape (T, J, 3).

    :param file_path: Path to the CSV file
    :return: NumPy array of shape (T, J, 3)
    """
    joint_data = []  # List to store frame-wise joint coordinates

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            frame_id = int(row[0])  # Extract frame index (not stored)
            joints = [ast.literal_eval(joint) for joint in row[1:]]  # Convert string to list
            joint_data.append(joints)

    joint_data = np.array(joint_data)  # Convert list to NumPy array
    if (np.size(joint_data, 0) > 1800):
        joint_data = joint_data[:1800, :, :]

    #for t in range(1800):
     #   for j in range(17):
      #      if (joint_data[t,j,0]**2 + joint_data[t,j,1]**2 == 0):
       #         print (file_path," ",t," ",j)

    return joint_data


def read_GT2D_csv(file_path, id=True):

    """
    Reads a CSV file and extracts joint coordinates into a NumPy array of shape (T, J, 3).

    :param file_path: Path to the CSV file
    :return: NumPy array of shape (T, J, 3)
    Since there is only two dims, we add the confidence as the third one, and initialized at 1.
    """
    joint_data = []  # List to store frame-wise joint coordinates

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            if id: st = 1
            else: st = 0
            joints = np.array([ast.literal_eval(joint) for joint in row[st:]])  # Convert string to list
            joints = np.reshape(joints, (-1, 2))
            joints_plus_confidence = np.ones((np.size(joints,0),3))
            joints_plus_confidence[:,:2] = joints
            joint_data.append(joints_plus_confidence)

    joint_data = np.array(joint_data)  # Convert list to NumPy array
    if (np.size(joint_data,0)>1800):
        joint_data = joint_data[:1800,:,:]

    return joint_data

def readGT2D_2(file_path):
    joint_data_left, joint_data_right = [], []  # List to store frame-wise joint coordinates

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            data = row[0]
            data = data.split(";")
            joints = [float(x) for x in data[1:]]
            joints_left = np.array(joints[:len(joints) // 2])
            joints_right = np.array(joints[len(joints) // 2:])
            joints_left = np.reshape(joints_left, (-1, 2))
            joints_right = np.reshape(joints_right, (-1, 2))
            # Supprimer le deuxième joint (cou) dans chaque matrice
            joints_left = np.delete(joints_left, 1, axis=0)
            joints_right = np.delete(joints_right, 1, axis=0)
            joint_data_left.append(joints_left)
            joint_data_right.append(joints_right)

    joint_data_left = np.array(joint_data_left)
    joint_data_right = np.array(joint_data_right)  # Convert list to NumPy array
    # if (np.size(joint_data, 0) > 1800):
    #     joint_data = joint_data[:1800, :, :]

    return joint_data_left, joint_data_right

def get_file_list_and_names(data_root, type_of_files, subset_list = None):

    file_list = []
    names_list = []
    if subset_list is None:
        subset_list = []
    for subset in subset_list:
        data_path = os.path.join(data_root, subset)
        patient_list = sorted(os.listdir(data_path))
        for patient in patient_list:
            patient_path = os.path.join(data_path, patient)
            patient_files = os.listdir(patient_path)
            patient_files = [x for x in patient_files if x.endswith(type_of_files)]
            names_list.append(patient)
            file_list.append(os.path.join(patient_path, patient_files[0]))

    return file_list, names_list

def save_patient_scores(patient_data, list_of_scores=["mpjpe", "p-mpjpe"], filename="patient_scores.csv"):

    """
    Saves patient names and scores to a CSV file.
    :param patient_data: List of tuples containing patient name and score
    :param list_of_scores: Names of scores to save
    :param filename: Name of the CSV file (default: patient_scores.csv)
    """

    if (os.path.exists(filename)):
        my_score_list = ["", ""] + ["" for i in range(len(list_of_scores))]
    else:
        my_score_list = ["Patient Name", "subset"] + list_of_scores
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(my_score_list)  # Writing header
        writer.writerows(patient_data)

def compute_scaling_factors(GT_2D, GT_3D):

    '''This function computes and returns the matrix of scaling factors s to go from pixel to camera coordinates(mm).
    GT_2D : pixel coordinates
    GT_3D : camera coordinates in mm'''

    s = np.zeros((GT_3D.shape[0], 2))
    real_rect_height = 600 ##in millimeters
    real_rect_width = 300
    for t in range(GT_3D.shape[0]):
        bbox_x_min = np.min(GT_2D[t, :, 0])  # Minimum x-coordinate across all joints
        bbox_x_max = np.max(GT_2D[t, :, 0])  # Maximum x-coordinate across all joints
        bbox_y_min = np.min(GT_2D[t, :, 1])  # Minimum y-coordinate across all joints
        bbox_y_max = np.max(GT_2D[t, :, 1])  # Maximum y-coordinate across all joints

        # Compute the bounding box size
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min

        ratiox = (bbox_width +1) / real_rect_width ##in pixels per millimeter
        ratioy = (bbox_height + 1) / real_rect_height  ##in pixels per millimeter
        s[t, 0] = 1 / ratiox
        s[t, 1] = 1 / ratioy
    return s ##in mm per pixels

def _infer_box(pose3d, camera, rect_size, rootIdx = 0):

    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] = tl_joint[:2] - rect_size / 2
    br_joint = root_joint.copy()
    br_joint[:2] = br_joint[:2] + rect_size / 2
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):

    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def h36m2coco(x):

    '''
        Convert Human3.6M keypoints (T x 17 x C) to COCO keypoints (T x 17 x C).

        Human3.6M keypoints order:
        0: "Pelvis", 1: "Right Hip", 2: "Right Knee", 3: "Right Ankle",
        4: "Left Hip", 5: "Left Knee", 6: "Left Ankle", 7: "Spine",
        8: "Neck", 9: "Nose", 10: "Head", 11: "Left Shoulder",
        12: "Left Elbow", 13: "Left Wrist", 14: "Right Shoulder",
        15: "Right Elbow", 16: "Right Wrist"

        COCO keypoints order:
        0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
        5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
        9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
        13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
    '''
    T, V, C = x.shape
    y = np.zeros([T, 17, C])

    # Mapping Human3.6M to COCO format
    y[:, 0, :] = x[:, 9, :]  # Nose
    y[:, 1, :] = x[:, 10, :]  # Left Eye (Approximated as Head)
    y[:, 2, :] = x[:, 10, :]  # Right Eye (Approximated as Head)
    y[:, 3, :] = x[:, 11, :]  # Left Ear (Approximated as Left Shoulder)
    y[:, 4, :] = x[:, 14, :]  # Right Ear (Approximated as Right Shoulder)
    y[:, 5, :] = x[:, 11, :]  # Left Shoulder
    y[:, 6, :] = x[:, 14, :]  # Right Shoulder
    y[:, 7, :] = x[:, 12, :]  # Left Elbow
    y[:, 8, :] = x[:, 15, :]  # Right Elbow
    y[:, 9, :] = x[:, 13, :]  # Left Wrist
    y[:, 10, :] = x[:, 16, :]  # Right Wrist
    y[:, 11, :] = x[:, 4, :]  # Left Hip
    y[:, 12, :] = x[:, 1, :]  # Right Hip
    y[:, 13, :] = x[:, 5, :]  # Left Knee
    y[:, 14, :] = x[:, 2, :]  # Right Knee
    y[:, 15, :] = x[:, 6, :]  # Left Ankle
    y[:, 16, :] = x[:, 3, :]  # Right Ankle

    return y

def image2camera(preds_3D, GT_3D, GT_2D, res_w, res_h):

    #Denormalization
    preds_3D[:, :, :2] = (preds_3D[:, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
    preds_3D[:, :, 2:] = preds_3D[:, :, 2:] * res_w / 2

    # We put the pelvis in the center
    preds_3D = preds_3D - preds_3D[:, 0:1, :]
    GT_3D = GT_3D - GT_3D[:, 0:1, :]
    GT_2D = GT_2D - GT_2D[:, 0:1, :]

    # We compute the scaling factors and change from pixel coordinates to camera coordinates
    scaling_factors = compute_scaling_factors(GT_2D, GT_3D)
    for t in range(np.size(GT_3D, 0)):
        for j in range(1, np.size(GT_3D, 1)):
            preds_3D[t, j, :] = preds_3D[t, j, :] * scaling_factors[t, j - 1]

    return preds_3D

def save_loss (file_path, loss_list, loss_title, x_title, x, n=1):

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.set_xlabel(x_title)
    if n == 1:
        ax.set_ylabel(loss_title)
        ax.plot(x, loss_list)
    else:
        ax.set_ylabel("loss")
        for i in range(n):
            ax.plot(x, loss_list[i], label=loss_title[i])
        ax.legend(loc='upper right')
    fig.savefig(file_path)  # save the figure to file
    plt.close(fig)  # close the figure window

def compute_scaling_factors_2(GT_2D, GT_3D, id_ref):

    '''This function computes and returns the matrix of scaling factors s to go from pixel to camera coordinates(mm).
    GT_2D : pixel coordinates
    GT_3D : camera coordinates in mm'''

    s = np.zeros((GT_3D.shape[0], 2))
    for t in range(GT_3D.shape[0]):
        denom = GT_2D[t, id_ref, 0]
        if denom == 0 or not np.isfinite(denom):
            denom = 300 / 720
        s[t, 0] = GT_3D[t, id_ref, 0] / denom
        s[t, 1] = GT_3D[t, id_ref, 1] / GT_2D[t, id_ref, 1]
    return s

def compute_uniform_scaling_factor(GT_2D, real_height_mm=600):

    '''Computes a single uniform scale factor (pixels → mm) per frame based on bounding box height.'''
    num_frames = GT_2D.shape[0]
    s = np.ones((num_frames,))
    for t in range(num_frames):
        y_coords = GT_2D[t, :, 1]
        bbox_height = np.max(y_coords) - np.min(y_coords)
        s[t] = real_height_mm / (bbox_height + 1e-5)  # Avoid division by zero
    return s

def compute_scaling_factors_lambda(joint_3d_image, joint_3d_cam, root_index):

    s = np.zeros((joint_3d_image.shape[0],))
    for t in range(joint_3d_image.shape[0]):
        x, y = joint_3d_image[t, root_index, 0], joint_3d_image[t, root_index, 1]
        X, Y = joint_3d_cam[t, root_index, 0], joint_3d_cam[t, root_index, 1]
        s[t] = (x*X + y*Y) / (x**2 + y**2)
    return s

def camera_to_image_frame(pose3d, camera, rootIdx, rectangle_3d_size):

    pose3d_image_frame = np.zeros_like(pose3d)
    ratios = np.zeros((pose3d.shape[0], ))
    for t in range(pose3d.shape[0]):
        box = _infer_box(pose3d[t, :, :], camera, rectangle_3d_size, rootIdx)
        ratio = (box[2] - box[0] + 1) / rectangle_3d_size
        pose3d_image_frame[t, :, :2] = _weak_project(
            pose3d[t, :, :].copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
        pose3d_depth = ratio * (pose3d[t, :, 2] - pose3d[t, rootIdx, 2])
        pose3d_image_frame[t, :, 2] = pose3d_depth
        ratios[t] = ratio
    return pose3d_image_frame, ratios

def normalize (pose, res_w, res_h, type="2d"):

    pose_norm = np.zeros_like(pose)
    pose_norm[:, :, 0] = 2 * pose[:, :, 0] / res_w - 1
    pose_norm[:, :, 1] = 2 * pose[:, :, 1] / res_w - (res_h / res_w)

    if type == "2d":
        pose_norm[:, :, 2] = pose[:, :, 2]
    else:
        pose_norm[:, :, 2] = 2 * pose[:, :, 2] / res_w

    return pose_norm

def denormalize (pose, res_w, res_h, mytype="2d"):

    pose_denorm = np.zeros_like(pose)
    pose_denorm[:, :, 0] = (pose[:, :, 0] + 1) * 0.5 * res_w
    pose_denorm[:, :, 1] = (pose[:, :, 1] + res_h / res_w) * 0.5 * res_w

    if mytype == "2d":
        pose_denorm[:, :, 2] = pose[:, :, 2]
    else:
        pose_denorm[:, :, 2] = pose[:, :, 2] * 0.5 * res_w

    return pose_denorm

def add_noise_and_conf(data, mu=0, sigma=0.3):
    noise = np.random.normal(loc=mu, scale=sigma, size=data.shape)
    data_with_noise = np.zeros((data.shape[0], data.shape[1], 3))
    data_with_noise[:, :, :2] = data + noise
    a, b, m, s = 1., -0.5, 0., 0.05
    for t in range(data.shape[0]):
        for j in range(data.shape[1]):
            dis = np.linalg.norm(data_with_noise[t, j, :2] - data[t, j, :2])
            conf = a / (dis + a) + b * dis
            shift = s * random.random() + m
            data_with_noise[t, j, 2] = np.array([conf + shift]).clip(0, 1)[0]
    return data_with_noise


def add_noise_and_conf_torch(data, mu=0.0, sigma=0.05, a=1.0, b=-0.5, m=0.0, s=0.05):
    """
    Add Gaussian noise and compute confidence based on displacement.
    Supports (N, T, J, C) and (N, V, T, J, C).
    """
    original_shape = data.shape
    has_views = len(original_shape) == 5  # (N, V, T, J, C)

    if has_views:
        N, V, T, J, C = original_shape
        data_reshaped = data[:, 0, ...]  # shape (N, T, J, C), assume all views are same for noise
    else:
        N, T, J, C = original_shape
        data_reshaped = data

    # Add noise only on first 2 coordinates
    noise = torch.randn_like(data_reshaped[..., :2]) * sigma + mu
    displacement = torch.norm(noise, dim=-1)  # (N, T, J)

    conf = a / (displacement + a) + b * displacement
    shift = torch.rand_like(conf) * s + m
    conf = (conf + shift).clamp(0, 1)

    # Construct output with noise and confidence
    noisy_2d = data_reshaped[..., :2] + noise
    data_with_noise = torch.zeros((N, T, J, 3), dtype=data.dtype, device=data.device)
    data_with_noise[..., :2] = noisy_2d
    data_with_noise[..., 2] = conf

    if has_views:
        data_with_noise = data_with_noise.unsqueeze(1).repeat(1, V, 1, 1, 1)

    return data_with_noise




def add_mask_numpy(x, mask_ratio, mask_T_ratio):
    """
    Args:
        x: np.ndarray of shape (T, J, C)
        mask_ratio: probability of masking a joint
        mask_T_ratio: probability of masking a frame

    Returns:
        x: masked np.ndarray of shape (T, J, C)
    """
    T, J, C = x.shape

    # Generate joint-wise mask (True = keep, False = mask)
    mask = (np.random.rand(T, J, 1) > mask_ratio).astype(x.dtype)

    # Generate temporal mask (True = keep, False = mask)
    mask_T = (np.random.rand(T, 1, 1) > mask_T_ratio).astype(x.dtype)

    # Apply both masks using broadcasting
    x_masked = x * mask * mask_T

    return x_masked

def add_mask_torch(x, mask_ratio, mask_T_ratio):
    """
    Random masking with fixed mask shared across views.
    Supports (N, T, J, C) and (N, V, T, J, C).
    """
    original_shape = x.shape
    has_views = len(original_shape) == 5  # (N, V, T, J, C)

    if has_views:
        N, V, T, J, C = x.shape
        x_reshaped = x[:, 0, ...]  # Take one view for mask generation
    else:
        N, T, J, C = x.shape
        x_reshaped = x

    mask = torch.rand(N, T, J, 1, dtype=x.dtype, device=x.device) > mask_ratio
    mask_T = torch.rand(1, T, 1, 1, dtype=x.dtype, device=x.device) > mask_T_ratio
    full_mask = mask * mask_T  # (N, T, J, 1)

    if has_views:
        full_mask = full_mask.unsqueeze(1).repeat(1, V, 1, 1, 1)
    x = x * full_mask

    return x


def rotate_pose(a, theta):
    '''a: pose_2d TxJx2'''
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    for t in range(a.shape[0]):
        a[t, :, :] = np.transpose(rot_matrix @ np.transpose(a[t, :, :]))
    return a

def pck3d_x(pred, gt, x):
    '''
    Returns the average 3D PCK with threshold x over all frames.

    Parameters:
        pred: np.ndarray of shape (T, J, C). Predicted pose in millimeters.
        gt: np.ndarray of shape (T, J, C). Ground truth pose in millimeters.
        x: float. Threshold in millimeters
    '''

    all_3d_pck_s = []
    for t in range(pred.shape[0]):
        distances = np.linalg.norm(pred[t, :, :] - gt[t, :, :], axis=-1)
        distances_bool = distances <= x
        distances_bool = distances_bool.astype(int)
        all_3d_pck_s.append(np.mean(distances_bool))
    all_3d_pck_s = np.array(all_3d_pck_s)
    return np.mean(all_3d_pck_s)

def rotation_matrix_3d(alpha, beta, gamma):
    '''
    Returns rotated matrix according to angles alpha (yaw), beta (pitch), gamma (roll).

    Parameters:
        alpha: float, yaw angle (rotation around Z-axis) in radians
        beta: float, pitch angle (rotation around Y-axis) in radians
        gamma: float, roll angle (rotation around X-axis) in radians

    Returns:
        rotated_pose: np.ndarray of shape (J, 3), rotated 3D pose
    '''

    # Rotation around Z-axis (Yaw)
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0, 0, 1]
    ])

    # Rotation around Y-axis (Pitch)
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation around X-axis (Roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ])

    # Combined rotation: R = Rz * Ry * Rx (Z-Y-X order)
    R = Rz @ Ry @ Rx

    return R

def make_new_view_2d(motion_3d_cam, extrinsic_parameters):
    camera = {'fx': 532.8180541992188,
              'fy': 532.8180541992188,
              'cx': 639.40,
              'cy': 374.1215}
    root_idx = 5  # left shoulder in COCO format
    rect_size = 1000  # in millimeters

    T, J, _ = motion_3d_cam.shape
    motion_3d_cam_rot = np.ones_like(motion_3d_cam)
    homog_motion_3d_cam = np.ones((T, J, 4))
    homog_motion_3d_cam[:, :, :3] = motion_3d_cam

    for t in range(T):
        result = (extrinsic_parameters @ homog_motion_3d_cam[t, :, :].T).T[:, :3]
        result = np.squeeze(result)  # remove any trailing (1) dimension
        if result.ndim == 2 and result.shape[1] == 3: motion_3d_cam_rot[t, :, :] = result
        elif result.ndim == 3: motion_3d_cam_rot[t, :, :] = result[:, :, 0]
        else: raise ValueError(f"Unexpected shape after projection: {result.shape}")


    motion_2d = np.ones_like(motion_3d_cam)
    motion_3d_image_rot, _ = camera_to_image_frame(motion_3d_cam_rot, camera, root_idx, rect_size)
    for t in range(T):
        for j in range(J):
            motion_2d[t, j, :2] = motion_3d_image_rot[t, j, :2]
            if not (0 <= motion_2d[t, j, 0] <= 880 and 0 <= motion_2d[t, j, 1] <= 720): motion_2d[t, j, 2] = 0

    return motion_2d























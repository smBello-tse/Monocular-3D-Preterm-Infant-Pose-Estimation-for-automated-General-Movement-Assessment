import argparse
import csv
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import pandas as pd
from math import cos,sin,acos,atan,asin,pi,degrees, atan2
from ast import literal_eval
import math
from lib.utils.agma import *
from lib.utils.agma import h36m2coco

#limbSeq = [[2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12],[12, 13]]
limbSeq = [[5, 2], [2, 3], [3, 4], [5, 6], [6, 7], [2, 8], [8, 9] ,[8, 11], [9, 10], [11, 5], [11, 12], [12, 13]]

def plotti(vect, color, fig_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    xs = []
    ys = []
    zs = []
    for a in vect:
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
    ax.scatter(xs, ys, zs,color = color)
    fig.savefig(fig_path)
    plt.close(fig)

def subplots(right_elbow, right_hand, left_elbow, left_hand, fig_path):
    fig2 = plt.figure()
    #fig2.show()
    sub11 = fig2.add_subplot(2, 2, 1, projection='3d')
    sub12 = fig2.add_subplot(2, 2, 3, projection='3d')
    sub21 = fig2.add_subplot(2, 2, 2, projection='3d')
    sub22 = fig2.add_subplot(2, 2, 4, projection='3d')
    xs = []
    ys = []
    zs = []
    for a in right_elbow:
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
    sub11.scatter(xs, ys, zs, color="b")
    sub11.title.set_text('right elbow')
    sub11.set_xlabel('X')
    sub11.set_ylabel('Y')
    sub11.set_zlabel('Z')

    xs = []
    ys = []
    zs = []
    for a in right_hand:
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
    sub12.scatter(xs, ys, zs, color="b")
    sub12.title.set_text('right hand')
    sub12.set_xlabel('X')
    sub12.set_ylabel('Y')
    sub12.set_zlabel('Z')


    xs = []
    ys = []
    zs = []
    for a in left_elbow:
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
    sub21.scatter(xs, ys, zs, color="r")
    sub21.title.set_text('left elbow')
    sub21.set_xlabel('X')
    sub21.set_ylabel('Y')
    sub21.set_zlabel('Z')

    xs = []
    ys = []
    zs = []
    for a in left_hand:
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
    sub22.scatter(xs, ys, zs, color="r")
    sub22.title.set_text('left hand')
    sub22.set_xlabel('X')
    sub22.set_ylabel('Y')
    sub22.set_zlabel('Z')
    fig2.savefig(fig_path)
    plt.close(fig2)

def save_activation_map(activation_map, fig_path):
    # Generate theta (0 to π) and phi (0 to 2π)
    theta_vals = np.linspace(0, np.pi, 64)
    phi_vals = np.linspace(0, 2 * np.pi, 128)

    # Set up ticks (for readability, we choose a subset)
    theta_tick_indices = np.linspace(0, 63, 5, dtype=int)  # 5 evenly spaced ticks
    phi_tick_indices = np.linspace(0, 127, 5, dtype=int)

    theta_tick_labels = [f"{theta_vals[i]:.2f}" for i in theta_tick_indices]
    phi_tick_labels = [f"{phi_vals[i]:.2f}" for i in phi_tick_indices]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(activation_map, aspect='auto', cmap='viridis', origin='lower')
    plt.xlabel('Phi (0 to 2π)')
    plt.ylabel('Theta (0 to π)')

    # Set ticks
    plt.xticks(phi_tick_indices, phi_tick_labels)
    plt.yticks(theta_tick_indices, theta_tick_labels)

    plt.title('Activation Map with Angular Axes')
    plt.colorbar(label='Activation')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


class Joint:
    def __init__(self,coords,length):
        self.coords = coords
        self.coords_norm=[]
        #normalize and convert to spherical coordinates
        for z in range(length):
            self.coords_norm.append(self.coords[z] / np.linalg.norm(self.coords[z]))
        self.spherical_coords = [[atan2(i[1], i[0]) * 64 / pi if atan2(i[1], i[0]) > 0 else (atan2(i[1], i[0]) + 2 * pi) * 64 / pi,
             acos(i[2]) * 64 / pi] for i in self.coords_norm]

        activ_map = np.zeros((65, 129))
        #create activation map and put 1s for activations
        for coord in self.spherical_coords:
            if not (math.isnan(coord[1]) or math.isnan(coord[1])):
                activ_map[int(coord[1]), int(coord[0])] = 1

        self.activ_map = activ_map[0:64,0:128]
        non_zeros = np.transpose(np.nonzero(self.activ_map))
        filtered_spherical_coords = non_zeros * pi /64
        #convert back to cartesian coordinates
        self.cartesian_coords = [[sin(d[0])*cos(d[1]),sin(d[0])*sin(d[1]),cos(d[0])] for d in filtered_spherical_coords]
        self.stats()

    def stats(self):
        # polarization and var calculation
        vec_sum_i = sum([i[0] for i in self.cartesian_coords])
        vec_sum_j = sum([i[1] for i in self.cartesian_coords])
        vec_sum_k = sum([i[2] for i in self.cartesian_coords])
        vec_sum = np.array([vec_sum_i, vec_sum_j, vec_sum_k])
        self.mean_dir = vec_sum / np.linalg.norm(vec_sum).sum(axis=0)
        self.polarization = np.linalg.norm(vec_sum) / len(self.cartesian_coords)
        self.var = 1 - self.polarization

def rotation_matrix(sign, vector1, vector2):
    rot_axe = np.cross(vector1, vector2)
    if np.linalg.norm(rot_axe) == 0:
        rot_angle = 0
    else:
     rot_axe = rot_axe / np.linalg.norm(rot_axe)
     rot_angle = acos(np.dot(vector1, vector2))
    ax, ay, az = rot_axe[0], rot_axe[1], rot_axe[2]

    if sign == "+":
        s = sin(rot_angle)
        c = cos(rot_angle)
    if sign == "-":
        s = sin(-rot_angle)
        c = cos(-rot_angle)
    u = 1 - c
    return np.array([[ax * ax * u + c, ax * ay * u - az * s, ax * az * u + ay * s],
                     [ay * ax * u + az * s, ay * ay * u + c, ay * az * u - ax * s],
                     [az * ax * u - ay * s, az * ay * u + ax * s, az * az * u + c], ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/outputs", help="Path to the 3d poses.")
    opts = parser.parse_args()
    return opts


opts = parse_args()
og_path = opts.dir
m3d_dir = f"m3d/"
os.makedirs(m3d_dir, exist_ok=True)
# GT_3D_list, patients = get_file_list_and_names(og_path, "coords_3D.npy", subset_list=[subset])
#preds_3D_list, patients = get_file_list_and_names(og_path, f"{method3d}{view}_{data_augmentation}.npy",
                                                  #subset_list=[subset])
# preds_3D_list1, patient_train = get_file_list_and_names(files_path, f"lr_{lr}_batch_{batch_size}_no_freeze_epochs_{nb_epochs}.npy", subset_list = ["train"])
# preds_3D_list2, patient_test = get_file_list_and_names(og_path, f"epochs_{nb_epochs}.npy", subset_list=["test"])
preds_3D_list = [f for f in os.listdir(og_path) if f.endswith(".npy")]

results_dataf = pd.DataFrame(columns=['Video', 'left_elbow', 'right_elbow',
                                      'left_wrist', 'right_wrist', 'upper_joints', 'left_knee',
                                      'right_knee',
                                      'left_ankle', 'right_ankle', 'lower_joints', 'all_joints'])

vid_counter = 0
for vi in preds_3D_list:
    counter = 0
    # dataf = np.load(vi)
    #print(vi, patient)
    dataf = np.load(os.path.join(og_path, vi))
    patient = os.path.splitext(vi)[0]

    right_elbow_coords = []
    right_shld_coords = []
    right_hand_coords = []
    right_knee_coords = []
    right_ankle_coords = []
    left_elbow_coords = []
    left_shld_coords = []
    left_hand_coords = []
    left_knee_coords = []
    left_ankle_coords = []
    # print(dataf)
    while counter < dataf.shape[0]:
        # print(counter)
        # line = dataf.iloc[counter][:]
        # verts = np.array([ np.array(a) for a in line.apply(literal_eval)])
        verts = dataf[counter, :, :]

        right_should = verts[6, :]
        left_should = verts[5, :]
        right_hip = verts[12, :]
        left_hip = verts[11, :]
        right_elbow = verts[8, :]
        left_elbow = verts[7, :]
        right_hand = verts[10, :]
        left_hand = verts[9, :]
        right_knee = verts[14, :]
        right_ankle = verts[16, :]
        left_knee = verts[13, :]
        left_ankle = verts[15, :]

        ####################################Upper_right#######################################"""
        middle_hip = (right_hip + left_hip) / 2

        vect_l_sh_h = middle_hip - right_should
        vect_r_sh_h = middle_hip - left_should
        vect_l_r_product = np.cross(vect_r_sh_h, vect_l_sh_h)
        i = left_should - right_should
        k = vect_l_r_product
        j = np.cross(i, k)
        i = i / np.linalg.norm(i)
        k = k / np.linalg.norm(k)
        j = j / np.linalg.norm(j)

        trans_mat_right_shld = np.array(
            [np.append(i, 0), np.append(j, 0), np.append(k, 0), np.append(right_should, 1)])
        # coords_world = np.dot(np.append(j,1), trans_mat_right_shld)
        right_elbow_local_shld = np.dot(np.append(right_elbow, 1), np.linalg.inv(trans_mat_right_shld))

        right_upper_arm = right_should - right_elbow
        right_upper_arm = right_upper_arm / np.linalg.norm(right_upper_arm)
        rot_max = rotation_matrix("+", right_upper_arm, i)
        i_rot = np.dot(rot_max, i)
        if list(i_rot) == list(i):
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        else:
            rot_max = rotation_matrix("-", right_upper_arm, i)
            i_rot = np.dot(rot_max, i)
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        trans_mat_right_elbow = np.array(
            [np.append(i_rot, 0), np.append(j_rot, 0), np.append(k_rot, 0), np.append(right_elbow, 1)])
        right_hand_local_elbow = np.dot(np.append(right_hand, 1), np.linalg.inv(trans_mat_right_elbow))
        right_elbow_coords.append(right_elbow_local_shld[0:-1])
        right_hand_coords.append(right_hand_local_elbow[0:-1])

        ####################################Upper_left#######################################"""

        i = right_should - left_should
        k = vect_l_r_product
        j = -np.cross(i, k)
        i = i / np.linalg.norm(i)
        k = k / np.linalg.norm(k)
        j = j / np.linalg.norm(j)

        trans_mat_left_shld = np.array(
            [np.append(i, 0), np.append(j, 0), np.append(k, 0), np.append(left_should, 1)])
        left_elbow_local_shld = np.dot(np.append(left_elbow, 1), np.linalg.inv(trans_mat_left_shld))
        left_upper_arm = left_should - left_elbow
        left_upper_arm = left_upper_arm / np.linalg.norm(left_upper_arm)
        rot_max = rotation_matrix("+", left_upper_arm, i)
        i_rot = np.dot(rot_max, i)
        if list(i_rot) == list(i):
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        else:
            rot_max = rotation_matrix("-", left_upper_arm, i)
            i_rot = np.dot(rot_max, i)
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        trans_mat_left_elbow = np.array(
            [np.append(i_rot, 0), np.append(j_rot, 0), np.append(k_rot, 0), np.append(left_elbow, 1)])
        left_hand_local_elbow = np.dot(np.append(left_hand, 1), np.linalg.inv(trans_mat_left_elbow))
        left_elbow_coords.append(left_elbow_local_shld[0:-1])
        left_hand_coords.append(left_hand_local_elbow[0:-1])
        ############################################lower_right############################################

        middle_should = (left_should + right_should) / 2
        vect_l_h_sh = middle_should - left_hip
        vect_r_h_sh = middle_should - right_hip
        vect_l_r_product = np.cross(vect_l_h_sh, vect_r_h_sh)

        i = left_hip - right_hip
        k = -vect_l_r_product
        j = np.cross(i, k)

        i = i / np.linalg.norm(i)
        k = k / np.linalg.norm(k)
        j = j / np.linalg.norm(j)

        trans_mat_right_hip = np.array(
            [np.append(i, 0), np.append(j, 0), np.append(k, 0), np.append(right_hip, 1)])
        # coords_world = np.dot(np.append(j, 1), trans_mat_right_hip)
        right_knee_local_hip = np.dot(np.append(right_knee, 1), np.linalg.inv(trans_mat_right_hip))

        right_upper_leg = right_hip - right_knee

        right_upper_leg = right_upper_leg / np.linalg.norm(right_upper_leg)
        rot_max = rotation_matrix("+", right_upper_leg, i)
        i_rot = np.dot(rot_max, i)
        if list(i_rot) == list(i):
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        else:
            rot_max = rotation_matrix("-", right_upper_leg, i)
            i_rot = np.dot(rot_max, i)
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        trans_mat_right_knee = np.array(
            [np.append(i_rot, 0), np.append(j_rot, 0), np.append(k_rot, 0), np.append(right_knee, 1)])
        right_ankle_local_knee = np.dot(np.append(right_ankle, 1), np.linalg.inv(trans_mat_right_knee))
        right_knee_coords.append(right_knee_local_hip[0:-1])
        right_ankle_coords.append(right_ankle_local_knee[0:-1])

        ############################################lower_left############################################
        i = right_hip - left_hip
        k = vect_l_r_product
        j = np.cross(i, k)

        i = i / np.linalg.norm(i)
        k = k / np.linalg.norm(k)
        j = j / np.linalg.norm(j)

        trans_mat_left_hip = np.array(
            [np.append(i, 0), np.append(j, 0), np.append(k, 0), np.append(left_hip, 1)])
        # coords_world = np.dot(np.append(j, 1), trans_mat_right_hip)
        left_knee_local_hip = np.dot(np.append(left_knee, 1), np.linalg.inv(trans_mat_left_hip))

        left_upper_leg = left_hip - left_knee
        left_upper_leg = left_upper_leg / np.linalg.norm(left_upper_leg)
        rot_max = rotation_matrix("+", left_upper_leg, i)
        i_rot = np.dot(rot_max, i)
        if list(i_rot) == list(i):
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        else:
            rot_max = rotation_matrix("-", left_upper_leg, i)
            i_rot = np.dot(rot_max, i)
            j_rot = np.dot(rot_max, j)
            k_rot = np.dot(rot_max, k)
        trans_mat_left_knee = np.array(
            [np.append(i_rot, 0), np.append(j_rot, 0), np.append(k_rot, 0), np.append(left_knee, 1)])
        left_ankle_local_knee = np.dot(np.append(left_ankle, 1), np.linalg.inv(trans_mat_left_knee))
        left_knee_coords.append(left_knee_local_hip[0:-1])
        left_ankle_coords.append(left_ankle_local_knee[0:-1])

        counter += 1
    right_Elbow = Joint(right_elbow_coords, dataf.shape[0])
    left_Elbow = Joint(left_elbow_coords, dataf.shape[0])
    right_Hand = Joint(right_hand_coords, dataf.shape[0])
    left_Hand = Joint(left_hand_coords, dataf.shape[0])
    right_Knee = Joint(right_knee_coords, dataf.shape[0])
    right_Ankle = Joint(right_ankle_coords, dataf.shape[0])
    left_Knee = Joint(left_knee_coords, dataf.shape[0])
    left_Ankle = Joint(left_ankle_coords, dataf.shape[0])

    Upper_joints_var = (right_Elbow.var + left_Elbow.var + right_Hand.var + left_Hand.var) / 4
    lower_joints_var = (right_Knee.var + left_Knee.var + right_Ankle.var + left_Ankle.var) / 4
    joints_var = (lower_joints_var + Upper_joints_var) / 2

    print(Upper_joints_var, lower_joints_var, joints_var)

    results_dataf.loc[len(results_dataf.index)] = [patient, left_Elbow.var, right_Elbow.var,
                                                   left_Hand.var,
                                                   right_Hand.var, Upper_joints_var, left_Knee.var,
                                                   right_Knee.var,
                                                   left_Ankle.var, right_Ankle.var, lower_joints_var,
                                                   joints_var]
    print(vid_counter, " ", vi)
    vid_counter += 1
    # data_norm = np.concatenate([np.array(right_Elbow.coords_norm),
    #                     np.array(left_Elbow.coords_norm),
    #                     np.array(right_Hand.coords_norm),
    #                     np.array(left_Hand.coords_norm),
    #                     np.array(right_Knee.coords_norm),
    #                     np.array(left_Knee.coords_norm),
    #                     np.array(right_Ankle.coords_norm),
    #                     np.array(left_Ankle.coords_norm)], axis=1)


    os.makedirs(os.path.join(m3d_dir + "maps"), exist_ok=True)
    os.makedirs(os.path.join(m3d_dir + "visu"), exist_ok=True)
    save_activation_map(right_Elbow.activ_map, os.path.join(m3d_dir + "maps", patient + ".png"))
    plotti(right_Elbow.coords_norm, "red", os.path.join(m3d_dir + "visu", patient + ".png"))
os.makedirs(m3d_dir + "evals", exist_ok=True)
# Normalize variance columns
variance_columns = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                    'upper_joints', 'left_knee', 'right_knee', 'left_ankle',
                    'right_ankle', 'lower_joints', 'all_joints']

results_dataf.to_csv(os.path.join(m3d_dir + "evals", "m3d.csv"), index=False)

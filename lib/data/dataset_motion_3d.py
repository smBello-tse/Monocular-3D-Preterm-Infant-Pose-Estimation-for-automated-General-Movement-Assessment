import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
from lib.utils.agma import *

def sampling_new_cameras(nb_new_cams, center, a, protocol="spherical", bimodal=False):
    '''This function returns the coordinates of new cameras, sampled in a sphere centered around center.

    parameters:
    nb_new_cams: number of new cameras
    cam_ref: coordinates of reference camera
    center: center of the sphere
    a : z limits
    '''

    #Changing coordinate system R0(x, y, z) => R_center(x_center, y_center, z_center)
    #cam_ref_center = cam_ref - center

    new_cam_list = []

    nb_attempts = nb_new_cams * 2000
    attempts = 0

    if protocol == "spherical":
        while len(new_cam_list) < nb_new_cams and attempts < nb_attempts:
            # Sampling spherical coordinates theta and phi, and z_center.
            phi = np.random.uniform(0, 2 * np.pi )
            # t = np.random.uniform()
            theta = np.random.uniform(np.pi / 2 + np.pi / 4, np.pi)
            # t = np.random.uniform()
            z_center = np.random.uniform(-a + 100, -a - 300)
            radius = z_center / np.cos(theta)

            if radius <= 0:
                attempts += 1
                continue

            # Computing cartesian coordinates in R_center
            new_cam = np.zeros((3,))
            new_cam[0] = radius * np.sin(theta) * np.cos(phi)
            new_cam[1] = radius * np.sin(theta) * np.sin(phi)
            new_cam[2] = z_center

            # Changing coordinate system  R_center => R_0
            new_cam = new_cam + center
            new_cam_list.append(new_cam)

    elif protocol == "cylindrical":
        while len(new_cam_list) < nb_new_cams and attempts < nb_attempts:

            if bimodal:
                t = np.random.normal(0, 1)
                z_center = np.random.normal(-a + 100, 50) if t < 0.3 else np.random.normal(-a - 300, 50)
            else:
                z_center = np.random.uniform(-a + 100, -a - 300)
            theta = np.random.uniform(np.pi / 2 + np.pi / 8, np.pi)
            r = z_center / np.cos(theta)
            rho = r * np.sin(theta)
            phi = np.random.uniform(0 , 2 * np.pi)

            # Computing cartesian coordinates in R_center
            new_cam = np.zeros((3,))
            new_cam[0] = rho * np.cos(phi)
            new_cam[1] = rho * np.sin(phi)
            new_cam[2] = z_center

            # Changing coordinate system  R_center => R_0
            new_cam = new_cam + center
            new_cam_list.append(new_cam)

    new_cam_list = np.array(new_cam_list)
    new_cam_list = np.reshape(new_cam_list, (-1, 3))
    return new_cam_list

def compute_extrinsic_parameters(cameras, directions):
    #Suivant le cours de Robert COLLINS, CSE486, Penn State, Lecture 12: Camera Projection.

    extrinsic_parameters, axis = [], []
    nb = cameras.shape[0]

    for n in range(nb):
        R = np.eye(4)
        T = np.eye(4)

        #Translation by -C
        T[0, 3] = -cameras[n, 0]
        T[1, 3] = -cameras[n, 1]
        T[2, 3] = -cameras[n, 2]

        #Rotation
        z_vector = directions[n, :] / np.linalg.norm(directions[n, :])
        R[:3, 2] = z_vector
        y_vector = np.cross(z_vector, np.array([1, 0, 0]))
        y_vector = y_vector / np.linalg.norm(y_vector)
        R[:3, 1] = y_vector
        x_vector = np.cross(y_vector, z_vector)
        x_vector = x_vector / np.linalg.norm(x_vector)
        axis.append(R)
        R[:3, 0] = x_vector
        R = R.T
        extrinsic_parameters.append(R @ T)
    return extrinsic_parameters, axis
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d
        self.noise = args.noise
        self.rotation = args.rotation
        self.curriculum_epoch = 0 #This will help us gradually add new views inside the training overtime
        self.scheduler = args.scheduler
        self.tau = args.tau
        if self.scheduler == "undefined": print("No scheduler defined!")

    def set_curriculum_epoch(self, epoch):
        self.curriculum_epoch = epoch

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]
        epoch = self.curriculum_epoch
        generate_new_views = True
        if self.scheduler == "linear":
            motion_3d_cam = motion_file["data_cam"]
            p = min(1.0, epoch / self.tau)  # Linearly ramp up p from 0 to 1 over 10 epochs
        elif self.scheduler == "exp":
            motion_3d_cam = motion_file["data_cam"]
            p = 1. - np.exp(-epoch / self.tau)    
        else:
            # print("No scheduler defined!")
            p = 0.
            generate_new_views = False
        #print(motion_3d_cam[0])
        T, J, _ = motion_3d.shape
        was_flipped = False
        

        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection
                motion_2d = motion_file["data_input"]
                t = random.random()
                if t < p:
                    N = 1
                    TARGET = np.mean(motion_3d_cam[0, :, :], axis=0 ) # The point the camera is looking at. In our case, it is the barycenter of the pose in the first frame.
                    a = TARGET[2]
                    positions= sampling_new_cameras(N, TARGET, a, "spherical", False)
                    directions = TARGET - positions
                    extrinsics = np.stack(compute_extrinsic_parameters(positions, directions))
                    motion_2d = make_new_view_2d(motion_3d_cam, extrinsics[0])
                    motion_2d = normalize(motion_2d, 880, 720)
                if self.flip and t < 0.5:
                    motion_2d = flip_data(motion_2d)
                    was_flipped = True
                    motion_3d = flip_data(motion_3d)

            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        if generate_new_views:
            motion_2d_ref = motion_file["data_ref"]
            if was_flipped: motion_2d_ref = flip_data(motion_2d_ref)
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), torch.FloatTensor(motion_2d_ref)
        else: return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
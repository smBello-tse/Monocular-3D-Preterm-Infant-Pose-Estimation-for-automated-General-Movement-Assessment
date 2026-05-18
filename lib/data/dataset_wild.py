import torch
import numpy as np
import math
from torch.utils.data import Dataset
from lib.utils.utils_data import crop_scale
from lib.utils.agma import *
from lib.data.dataset_motion_3d import sampling_new_cameras, compute_extrinsic_parameters


def coco2h36m(x):
    '''
        Convert COCO keypoints (T x 17 x C) to Human3.6M keypoints (T x 17 x C).
        COCO keypoints order:
        0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
        5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
        9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
        13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"

        Human3.6M keypoints order:
        0: "Pelvis", 1: "Right Hip", 2: "Right Knee", 3: "Right Ankle",
        4: "Left Hip", 5: "Left Knee", 6: "Left Ankle", 7: "Spine",
        8: "Neck", 9: "Nose", 10: "Head", 11: "Left Shoulder",
        12: "Left Elbow", 13: "Left Wrist", 14: "Right Shoulder",
        15: "Right Elbow", 16: "Right Wrist"
    '''
    T, V, C = x.shape
    y = np.zeros([T, 17, C])

    # Mapping COCO to Human3.6M format
    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) / 2  # Pelvis (Mid Hip)
    y[:, 1, :] = x[:, 12, :]  # Right Hip
    y[:, 2, :] = x[:, 14, :]  # Right Knee
    y[:, 3, :] = x[:, 16, :]  # Right Ankle
    y[:, 4, :] = x[:, 11, :]  # Left Hip
    y[:, 5, :] = x[:, 13, :]  # Left Knee
    y[:, 6, :] = x[:, 15, :]  # Left Ankle
    y[:, 7, :] = (x[:, 5, :] + x[:, 6, :]) / 2  # Spine (Mid Shoulder)
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) / 2  # Neck (Same as Spine)
    y[:, 9, :] = x[:, 0, :]  # Nose
    y[:, 10, :] = (x[:, 1, :] + x[:, 2, :]) / 2  # Head (Mid Eyes)
    y[:, 11, :] = x[:, 5, :]  # Left Shoulder
    y[:, 12, :] = x[:, 7, :]  # Left Elbow
    y[:, 13, :] = x[:, 9, :]  # Left Wrist
    y[:, 14, :] = x[:, 6, :]  # Right Shoulder
    y[:, 15, :] = x[:, 8, :]  # Right Elbow
    y[:, 16, :] = x[:, 10, :]  # Right Wrist

    return y



def read_input(input_path, vid_size):

    joint_2d = np.load(input_path)
    if C == 2:
        joint_2d = np.concatenate((joint_2d, np.ones((T, J, 1))), axis=2)
    joint_2d = coco2h36m(joint_2d)
    return normalize(joint_2d, vid_size[0], vid_size[1])


class WildDetDataset(Dataset):
    def __init__(self, input_path, clip_len=243, vid_size=None):
        self.input_path = input_path
        self.clip_len = clip_len
        self.vid_all = read_input(input_path, vid_size)
        self.vid_size = vid_size
        self.Frames2DCount = len(self.vid_all) if type(self.input_path) == str else len(self.vid_all[0])


    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(self.Frames2DCount / self.clip_len)

    def __getitem__(self, index):
        'Generates one sample of data'
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, self.Frames2DCount)
        T, J, C = np.shape(self.vid_all)
        return self.vid_all[ st:end, :, :]

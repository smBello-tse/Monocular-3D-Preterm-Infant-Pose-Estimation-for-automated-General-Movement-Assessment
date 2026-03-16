import os
import sys
import pickle
import numpy as np
import random
sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_h36m import DataReaderH36M
from tqdm import tqdm
# To convert sequences into clips of n_frames frames (Here, we used n_frames=81)


def save_clips(subset_name, root_path, train_data, train_labels, train_cam, train_ref):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label, data_cam, data_ref = train_data[i], train_labels[i], train_cam[i], train_ref[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label,
            "data_cam": data_cam,
            "data_ref": data_ref
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
            pickle.dump(data_dict, myprofile)
            
datareader = DataReaderH36M(n_frames=81, sample_stride=1, data_stride_train=81, data_stride_test=81, dt_file = 'data.pkl', dt_root='data/motion3d/')
train_data, test_data, train_labels, test_labels, train_factors, test_factors, train_cam, test_cam, train_data_ref, test_data_ref = datareader.get_sliced_data()
print(train_data.shape, test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

root_path = "data/motion3d/MB3D_f243s81/DATA-SH"
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels, train_cam, train_data_ref)
save_clips("test", root_path, test_data, test_labels, test_cam, test_data_ref)


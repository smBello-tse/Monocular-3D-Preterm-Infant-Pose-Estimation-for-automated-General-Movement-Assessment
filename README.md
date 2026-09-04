### INSTALLATION GUIDE

Computer specs:

- OS: WSL Ubuntu 22.04
- Python: 3.10.12
- CUDA: 12.6
- CUDA toolkit: 12.1

The first step is to install torch 2.10.0 and torchvision 0.25.0 for CUDA 12.6 (https://pytorch.org/get-started/locally/):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Next, install the libraries listed in the file ```requirements.txt```. The addition of ```--no-build-isolation``` was necessary to avoid errors like ```ModuleNotFoundError: No module named 'pip'```

```
pip install -r requirements.txt --no-build-isolation
```

and for PoseMamba, also run:
```
cd kernels/selective_scan/ 
pip install -e . --no-build-isolation
cd ../../
```

Finally, install pytorch3d:
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d/
pip install -e . --no-build-isolation
cd ../
```

### DATA PREPARATION

First, put your raw data (2D joints coordinates and 3d ground truth as np.array NxTxJxC, where N is the number of patients, T is the number of frames, J is the number of joints, and C is the number of coordinates) inside the directory called ```data```. It should look like this:
```
data/
├── raw_data/
│   ├── train/
│   │   ├── patient_1/
│   │   │   ├── 2d_pose.npy
│   │   │   └── 3d_pose.npy
│   │   ├── patient_2/
│   │   │   ├── 2d_pose.npy
│   │   │   └── 3d_pose.npy
│   │   └── ...
│   ├── test/
│   │   ├── patient_k/
│   │   │   ├── 2d_pose.npy
│   │   │   └── 3d_pose.npy
│   │   ├── patient_k+1/
│   │   │   ├── 2d_pose.npy
│   │   │   └── 3d_pose.npy
│   │   └── ...
│   └── infer/
│       ├── patient_x/
│       │   ├── 2d_pose.npy
│       │   └── 3d_pose.npy
│       ├── patient_x+1/
│       │   ├── 2d_pose.npy
│       │   └── 3d_pose.npy
│       └── ...

```

Next, run ```make2.5d.py``` to convert 3d ground truth to 2.5d space. Change the extrinsic parameters in variable ```camera``` to suit your settings. Note that this code assumes that poses are in millimeters and in COCO format.

```
python make2.5d.py
```

Finally, run ```convert2pkl.py``` then ```tools/convert_h36m.py```. In ```convert2pkl.py```, you will also need to specify the width and height of the original videos in variables ```res_w``` and ```res_h```, respectively, and the numbers of frames, joints, and 2D features(x and y, or x, y, and confidence) in variables ```nb_frames```, ```nb_joints```, and ```nb_2d_features```, respectively.
```
python convert2pkl.py
python tools/convert_h36m.py
```

The links to collect the data and checkpoints are and ,respectively.

### TRAINING AND VALIDATION

NB: Do not use ```persistent_workers```, as this will completely disable new view generation.

To train MotionBERT on a specific checkpoint, run:
```
python ./train_mb.py --config ./configs/MB_lite_train_h36m.yaml --checkpoint ./checkpoint/where_to_save_next_checkpoint --pretrained ./checkpoint/path_to_directory_where_your__checkpoint_is --selection your_checkpoint.bin
```
For example:
```
python train_mb.py --config configs/MB_lite_train_h36m.yaml --checkpoint checkpoint/ --pretrained checkpoint/mb/ --selection mb_mod.bin
```
For PoseMamba, use ```train_pml.py```, instead.

To run generate new view angle during training, change ```scheduler``` in the configuration file to either ```exp``` or ```linear```. To use our modified version of MotionBERT, change ```model_type``` in the configuration file to ```mod```. To use the original version, use the value ```og```.

### INFERENCE

Note that without groundtruth, 3D estimated pose will be in 2.5d coordinates. See, Ci, H., Wang, C., Ma, X., & Wang, Y. (2019). Optimizing network structure for 3d human pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 2262-2271).

For both MotionBERT and PoseMamba, run:
```
python inference.py --config configs/your_config.yaml -e checkpoint/path_to_checkpoint.bin --gt --save_test 
```

```--gt``` and ```--save_test``` are used indicate the presence of GT and compute MPJPE, and for saving results, respectively.

### GMA CLASSIFICATION

The first step for GMA classification is to compute the mean 3d dispersion (see Soualmi A, Alata O, Ducottet C, Patural H, Giraud A. Mean 3D Dispersion for Automatic General Movement Assessment of Preterm Infants. Annu Int Conf IEEE Eng Med Biol Soc. 2023 Jul;2023:1-5. doi: 10.1109/EMBC40787.2023.10340961. PMID: 38083633.) using ```evaluateM3D.py```. It creates three directories, one containing activation maps (maps), one containing 3D visualizations of normalized coordinates of the right elbow (visus), and one containing the values of M3D in a csv file (evals).
```commandline
python evaluateM3D.py --dir path_to_your_3d_poses
```

The second step is classification using ```classification.py``` and the gt file provided in this work: ```Decision.xlxs```. Both file must be placed at the same level in the file tree.
```commandline
python classification.py --dir path_to_m3d_result_file --dir name_of_m3d_result_file

CI in the output summary means "confidence interval."

### CREDITS AND ACKNOWLEGMENTS
This repository was built upon codes available at the repositories for Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., & Wang, Y. (2023). Motionbert: A unified perspective on learning human motion representations. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 15085-15099) (https://github.com/Walter0807/MotionBERT); Huang, Y., Liu, J., Xian, K., & Qiu, R. C. (2025, April). Posemamba: Monocular 3d human pose estimation with bidirectional global-local spatio-temporal state space model. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 4, pp. 3842-3850) (https://github.com/nankingjing/PoseMamba); and Ci, H., Wang, C., Ma, X., & Wang, Y. (2019). Optimizing network structure for 3d human pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 2262-2271) (https://github.com/CHUNYUWANG/lcn-pose).
```

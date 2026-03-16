import copy
import os
import numpy as np
import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset, coco2h36m
from lib.utils.agma import *
import sys
from lib.model.loss import *
from lib.data.dataset_motion_3d import sampling_new_cameras, compute_extrinsic_parameters
from train_mb import project_3d_to_2d
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_init.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--csv', type=str, help='2D annotations root')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixel coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=81, help='clip length for network input')
    parser.add_argument("--method3d", type=str, default="mb", help="Name of 3d method")
    parser.add_argument("--subset", type=str, default="test", help="Subset")
    parser.add_argument("--gt", action="store_true", help="If gt is present for evaluation")
    parser.add_argument("--save_test", action="store_true", help="Whether to save the test results as npy files")
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)


    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        print("cuda available")
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage, weights_only=False)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos = model_backbone
    model_pos.eval()

    testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,  # Change this from 8 to 0 for Windows compatibility
        'pin_memory': True,
        'drop_last': False
    }

    
    patients_dir = "data/raw_data/infer"
    patients_list = os.listdir(patients_dir)
    mpjpes = []
    root_idx = 11  # left shoulder in Human3.6m
    i = 0
    input_feat = 3 if args.backbone == "DSTformer" else 2

    for patient in patients_list:
        files_path = os.path.join(patients_dir, patient)
        files = os.listdir(files_path)
        vid_path = os.path.join(files_path, [f for f in files if f.endswith(".mp4")][0])
        cap = cv2.VideoCapture(vid_path)
        fps_in = int(cap.get(cv2.CAP_PROP_FPS))
        vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        input_path = os.path.join(files_path, [f for f in files if f.endswith(f"2d_pose.npy")][0])

        wild_dataset = WildDetDataset(input_path, clip_len=opts.clip_len, vid_size=vid_size)
        print("aligning with pixel coordinates")

        print(f"Dataset length: {len(wild_dataset)}")

        test_loader = DataLoader(wild_dataset, **testloader_params)

        results_all = []
        proj_all = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                if torch.cuda.is_available():
                    batch_input = batch_input.float().cuda()

                if args.model_type == "og":
                    predicted_3d_pos = model_pos(batch_input[:, :, :, :input_feat])
                else:
                    predicted_3d_pos = model_pos(batch_input[:, :, :, :input_feat])[0]
                results_all.append(predicted_3d_pos.cpu().numpy())
        max_length = max([arr.shape[1] for arr in results_all])  # Find the longest sequence
        padded_results = [np.pad(arr, ((0, 0), (0, max_length - arr.shape[1]), (0, 0), (0, 0)), mode='constant')
                            for
                            arr
                            in
                            results_all]

        results_all = np.concatenate(padded_results)

        print(f"Shape of results_all before transposing: {results_all.shape}")
        results_all = np.concatenate(results_all, axis=0)  # Merge along the time dimension
        preds_3D = results_all[:wild_dataset.Frames2DCount, :, :]
        preds_3D = denormalize(preds_3D, vid_size[0], vid_size[1], "3d")
        if opts.save_test:
            os.makedirs("data/outputs", exist_ok=True)
            np.save(f"data/outputs/{patient}.npy", preds_3D)
        else:
            print("npy not saved!")

        if opts.gt:
            gt_path = os.path.join(files_path, [f for f in os.listdir(files_path) if f.endswith("3d_pose.npy")][0])
            ratios_path = os.path.join(files_path, [f for f in os.listdir(files_path) if f.endswith("ratios.npy")][0]) 
            gt = np.load(gt_path)
            ratios = np.load(ratios_path)
            for t in range(preds_3D.shape[0]):
                ratio = ratios[t, 0]
                for j in range(preds_3D.shape[1]):
                    preds_3D[t, j, :] /= ratio
            gt = coco2h36m(gt[:, :, :3])
            preds_3D -= preds_3D[:, root_idx:root_idx + 1, :]
            gt -= gt[:, root_idx:root_idx + 1, :]
            mpjpes.append(mpjpe(preds_3D, gt).mean())
            if i == len(patients_list) - 1:
                csv_path = os.path.join("data/outputs", "mpjpe_results.csv")
                with open(csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['Patient', 'MPJPE'])
                    for patient_name, mpjpe_value in zip(patients_list, mpjpes):
                        csv_writer.writerow([patient_name, mpjpe_value])
                print(f"MPJPE results saved to {csv_path}")
        i += 1
                
        
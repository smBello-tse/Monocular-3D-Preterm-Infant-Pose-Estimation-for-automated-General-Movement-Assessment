import sys
import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.data.dataset_wild import coco2h36m
from lib.utils.agma import save_loss, add_noise_and_conf_torch, add_mask_torch, rotation_matrix_3d
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.model.loss import *
from pytorch3d.transforms import so3_exp_map as rodrigues
from lib.data.dataset_motion_3d import MotionDataset3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH',
                        help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME',
                        help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-pr', '--protocol', default=0, type=int, help='0 for training and 1 for pretraining')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss
    }, chk_path)


def project_3d_to_2d(predicted_3d_pos, proj_params):
    """
    Projects 3D joint coordinates to 2D using a projection matrix.

    Args:
        predicted_3d_pos: torch.Tensor of shape (N, T, J, 3)
        proj_params: torch.Tensor of shape (N, T, 6), alpha, beta, gamma, tx, ty, tz

    Returns:
        projected_2d: torch.Tensor of shape (N, T, J, 2)
    """
    N, T, J, _ = predicted_3d_pos.shape
    rot = torch.from_numpy(np.zeros((N, T, 3, 4)))
    rot[:, :, :, 3] = proj_params[:, :, 3:6]
    for t in range(T):
        rot[:, t, :, :3] = rodrigues(proj_params[:, t, :3])
    rot = rot.cuda().float()

    # Convert 3D points to homogeneous coordinates (add ones): (N, T, J, 4)
    ones = torch.ones_like(predicted_3d_pos[..., :1])  # (N, T, J, 1)
    homog = torch.cat([predicted_3d_pos, ones], dim=-1)  # (N, T, J, 4)

    # Perform matrix multiplication: (N, T, J, 3)
    proj = torch.matmul(rot.unsqueeze(2), homog.unsqueeze(-1)).squeeze(-1)

    # Normalize by depth (z) to get 2D projection: (N, T, J, 2)
    #proj = proj[..., :2] / proj[..., 2:].clamp(min=1e-8)

    return proj

def make_new_2d_views(p2d, p3d, nb_views):
    N, T, J, _ = p3d.shape
    device = p3d.device

    # Output tensor: (N, nb_views+1, T, J, 3)
    views = [p2d]  # original view (already with (x, y, conf))

    for _ in range(nb_views):
        # Generate random angles and translation
        alpha = random.choice([np.pi / 18, -np.pi / 18, np.pi / 36, -np.pi / 36])
        beta  = random.choice([np.pi / 18, -np.pi / 18, np.pi / 36, -np.pi / 36])
        gamma = random.choice([np.pi / 18, -np.pi / 18, np.pi / 36, -np.pi / 36])
        tx, ty, tz = np.random.uniform(-50, 50, size=3)

        # Rotation and translation matrix
        R = rotation_matrix_3d(alpha, beta, gamma)  # (3,3)
        t = np.array([[tx], [ty], [tz]])            # (3,1)
        P = np.concatenate((R, t), axis=1)          # (3,4)
        P = torch.tensor(P, dtype=p3d.dtype, device=device)  # to torch

        # Homogenize 3D pose: (N, T, J, 4)
        ones = torch.ones((N, T, J, 1), dtype=p3d.dtype, device=device)
        p3d_homo = torch.cat([p3d, ones], dim=-1)  # (N, T, J, 4)

        # Flatten and apply projection: (N*T*J, 4) x (4, 3)^T = (N*T*J, 3)
        p3d_flat = p3d_homo.view(-1, 4)             # (N*T*J, 4)
        p2d_proj_flat = torch.matmul(p3d_flat, P.T) # (N*T*J, 3)
        p2d_proj = p2d_proj_flat.view(N, T, J, 3)   # (N, T, J, 3)

        # Normalize x, y by z (perspective division)
        eps = 1e-6
        x = p2d_proj[..., 0]
        y = p2d_proj[..., 1]
        conf = p2d[..., 2]  # re-use original confidence

        new_view = torch.stack([x, y, conf], dim=-1)  # (N, T, J, 3)
        views.append(new_view)

    # Stack all views: (nb_views+1, N, T, J, 3) -> (N, nb_views+1, T, J, 3)
    views = torch.stack(views, dim=1)
    return views



def evaluate(args, model_pos, test_loader, datareader, losses=None, fusion_=False, model_view=None):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()
    if fusion_: model_view.eval()
    with torch.no_grad():
        for batches in tqdm(test_loader):
            if torch.cuda.is_available():
                batches = [b.cuda(non_blocking=True) for b in batches]
            if len(batches)==2: 
                batch_input, batch_gt = batches
                batch_ref = None
            else: batch_input, batch_gt, batch_ref = batches
            batch_size = len(batch_input)
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if fusion_:
                print("FUSION MODE")

                preds_init, rot = model_view(batch_input)
                preds_init[..., 2] = 1.
                new_batch_input = torch.stack([batch_input, preds_init], dim=1)
                predicted_3d_pos, rot = model_pos(new_batch_input)  # (N, T, 17, 3)
            else:
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    if args.model_type=="og":
                        predicted_3d_pos_1 = model_pos(batch_input)
                        predicted_3d_pos_flip = model_pos(batch_input_flip)
                    else:
                        predicted_3d_pos_1, rot1 = model_pos(batch_input)
                        predicted_3d_pos_flip, rot2 = model_pos(batch_input_flip)
                        rot = (rot1 + rot2) / 2
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
                else:
                    predicted_3d_pos, rot = model_pos(batch_input) #A nettoyer.

            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                batch_gt[:, 0, 0, 2] = 0

            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            if losses is not None:
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_lv = loss_limb_var(predicted_3d_pos)
                loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
                loss_a = loss_angle(predicted_3d_pos, batch_gt)
                loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
                loss_depth = loss_mpjpe(predicted_3d_pos[..., 2:3], batch_gt[..., 2:3])


                if not(args.model_type=="og"): proj = project_3d_to_2d(predicted_3d_pos, rot)
                if batch_ref is None: loss_reproj = 0
                else: loss_reproj = loss_mpjpe(predicted_3d_pos[...,:2], batch_ref[...,:2] - batch_ref[:, :, 11:12, :2])
                loss_view = 0. if args.model_type=="og" else loss_mpjpe(proj[..., :2], batch_input[...,:2])
                loss_total = loss_3d_pos + \
                             args.lambda_scale * loss_3d_scale + \
                             args.lambda_3d_velocity * loss_3d_velocity + \
                             args.lambda_lv * loss_lv + \
                             args.lambda_lg * loss_lg + \
                             args.lambda_a * loss_a + \
                             args.lambda_av * loss_av + args.lambda_reproj * loss_reproj + args.lambda_view * loss_view + args.lambda_depth * loss_depth 
                losses['total_validation'].update(loss_total.item(), batch_size)
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    # gts = coco2h36m(gts)
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        # print("factor: ", factor[0,...])
        gt = gt_clips[idx]
        pred = results_all[idx]
        #print("pred: ", pred[0,...])
        #print("gt: ", gt[0,...])
        pred *= factor

        # Root-relative Errors
        root_idx = 11
        pred = pred - pred[:, root_idx:root_idx + 1, :]  # The root is put in the center
        gt = gt - gt[:, root_idx:root_idx + 1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, results_all


def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt, fusion_ =False, model_view = None):
    model_pos.train()
    if fusion_: model_view.eval()
    for idx, batches in tqdm(enumerate(train_loader)):
        if torch.cuda.is_available():
            batches = [b.cuda(non_blocking=True) for b in batches]
        if len(batches)==2: 
            batch_input, batch_gt = batches
            batch_ref = None
        else: batch_input, batch_gt, batch_ref = batches
        batch_size = len(batch_input)
        device = batch_input.device            
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:, :, :, 2:])  # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 11:12, :]
            else:
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 11:12,
                                                              2]  # Place the depth of first frame root to 0.
            if args.noise:
                batch_input = add_noise_and_conf_torch(batch_input)
            if args.mask_ratio:
                batch_input = add_mask_torch(batch_input, args.mask_ratio, args.mask_T_ratio)
        # Predict 3D poses
        if fusion_:

            preds_init, rot = model_view(batch_input)
            preds_init[..., 2] = 1.
            new_batch_input = torch.stack([batch_input, preds_init], dim=1)
            predicted_3d_pos, rot = model_pos(new_batch_input)  # (N, T, 17, 3)
            proj = project_3d_to_2d(predicted_3d_pos, rot)


        else:
            if args.model_type=="og": predicted_3d_pos = model_pos(batch_input)
            else:            
                predicted_3d_pos, rot = model_pos(batch_input)  # (N, T, 17, 3)
                proj = project_3d_to_2d(predicted_3d_pos, rot)
                print("mod")

        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_depth = loss_mpjpe(predicted_3d_pos[..., 2:3], batch_gt[..., 2:3])
                # Device-safe joint weights
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4], device=device)
            loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)

                # Temporal consistency loss
            dif_seq = predicted_3d_pos[:, 1:, :, :] - predicted_3d_pos[:, :-1, :, :]
            weights_joints = torch.ones_like(dif_seq, device=device)
            weights_joints = weights_joints.permute(0, 1, 3, 2) * w_mpjpe
            weights_joints = weights_joints.permute(0, 1, 3, 2)
            loss_diff = torch.mean(weights_joints * dif_seq.pow(2))

            #loss_reproj_fn = nn.MSELoss()

            loss_view = 0. if args.model_type=="og" else loss_mpjpe(proj[..., :2], batch_input[..., :2])
            if batch_ref is None: loss_reproj = 0
            else: loss_reproj = loss_mpjpe(predicted_3d_pos[...,:2], batch_ref[...,:2] - batch_ref[:, :, 11:12, :2])
            loss_total = loss_3d_pos + \
                         args.lambda_scale * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv * loss_lv + \
                         args.lambda_lg * loss_lg + \
                         args.lambda_a * loss_a + \
                         args.lambda_av * loss_av + args.lambda_reproj * loss_reproj + args.lambda_view * loss_view + args.lambda_depth * loss_depth + args.lambda_diff * loss_diff
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
            print(f"loss_reproj: {loss_reproj}")
            print(f"loss_view: {loss_view}")
            print(f"loss_3d_pos: {loss_3d_pos}")
            print(f"loss_total: {loss_total}")
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()


def train_with_config(args, opts):
    print(args)
    avg_mpjpe_list, avg_p_mpjpe_list, epoch_list, loss_train, loss_validation = [], [], [], [], []
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'prefetch_factor': None,
        'persistent_workers': False
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
        'prefetch_factor': None,
        'persistent_workers': False
    }
    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    print("let's go")
    print("len(train_dataset)= ", len(train_dataset))

    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)

    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                dt_root='data/motion3d', dt_file=args.dt_file)
    min_loss = 100000
    print(args.backbone)
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()


    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
        model_pos = model_backbone

    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr,
                                weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d),
                                                                   len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']

        # args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        # if args.mask or args.noise:
        #    args.aug = Augmenter2D(args)
        my_lr = args.learning_rate
        my_dropout = args.dropout
        my_wd = args.weight_decay

        # eval test
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
        best_model, best_epoch, best_lr = model_pos, 0, lr
        min_val_loss = 100000000
        fusion_ = False

        # Training
        for epoch in range(st, args.epochs):
            avg_mpjpe_list.append(e1)
            avg_p_mpjpe_list.append(e2)
            epoch_list.append(epoch)
            save_loss(os.path.join(opts.checkpoint, "avg_mpjpe.png"), avg_mpjpe_list, "MPJPE (mm)", "epoch",
                      epoch_list)
            save_loss(os.path.join(opts.checkpoint, "avg_p_mpjpe.png"), avg_p_mpjpe_list, "P-MPJPE (mm)", "epoch",
                      epoch_list)
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            losses['total_validation'] = AverageMeter()
            N = 0

            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_dataset.set_curriculum_epoch(epoch)
            if fusion_: train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True, fusion_=fusion_, model_view=model_view)
            else: train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg))
            else:
                if fusion_: e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader, losses, fusion_, model_view)
                else: e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader, losses)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                train_writer.add_scalar('loss_total_validation', losses['total_validation'].avg, epoch + 1)
                loss_train.append(losses['total'].avg)
                loss_validation.append(losses['total_validation'].avg)
                save_loss(os.path.join(opts.checkpoint, "losses.png"), [loss_train, loss_validation],
                          ["loss_train", "loss_val"], "epoch",
                          epoch_list, n=2)

            # Decay learning rate exponentially

            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')

            #save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            #if (epoch + 1) % args.checkpoint_frequency == 0:
                #save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if epoch > 5 and min_loss < 2 and not fusion_:
                print("\n\n##########################################################################\nFUSION MODE")
                fusion_ = True
                model_view = best_model.cuda()
            if e1 < min_loss and loss_validation[-1] < min_val_loss:
                min_loss, min_val_loss = e1, loss_validation[-1]
                best_model, best_epoch, best_lr = model_pos, epoch, lr
                if args.save_best: save_checkpoint(os.path.join(opts.checkpoint, "best_epoch.bin"), best_epoch, best_lr, optimizer, best_model, min_loss)
            print(f"min_loss : {min_loss}, best_epoch : {best_epoch}")



        return min_loss

    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
        return e1


if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    e1 = train_with_config(args, opts)
    
#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import shutil
import json
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/lea/trampo/MODELS_2D3D/mmpose')

from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

from mmdet.apis import DetInferencer

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from utils import predict_multiview_with_grad, find_best_triangulation, project_points


class MultiViewDataset(Dataset):
    def __init__(self, root_dir, K, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Discover all sequences
        sequences = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.sequences = sorted(sequences)
        
        self.sequence_data = []  # Will hold (seq_name, frame_names, calibration)
        self.index_map = []      # Global index -> (seq_idx, frame_idx)

        for seq_idx, seq_name in enumerate(self.sequences):
            seq_path = os.path.join(root_dir, seq_name)

            # Load calibration for this sequence
            session = seq_name.split('-')[0].split('_')[2]
            calib_path = os.path.join('calib', f'WorldTCam_{session}.npz')

            world_T_cam = np.load(calib_path)['arr_0']
            projMat = np.stack([np.linalg.inv(mat) for mat in world_T_cam])

            Ts = torch.tensor(projMat, dtype=torch.float32)
            Ks = torch.tensor(K, dtype=torch.float32)

            # Get camera dirs
            cam_dirs = sorted([d for d in os.listdir(seq_path) if d.startswith("Cam")])
            cam_dirs = [os.path.join(seq_path, d) for d in cam_dirs]
            self.cam_dirs = cam_dirs

            # Get frames from first camera (assume sync)
            frame_names = sorted(os.listdir(cam_dirs[0]))

            # Store metadata for this sequence
            self.sequence_data.append({
                "name": seq_name,
                "cam_dirs": cam_dirs,
                "frame_names": frame_names,
                "Ks": Ks,
                "Ts": Ts
            })

            # Build global index mapping
            for frame_idx in range(len(frame_names)):
                self.index_map.append((seq_idx, frame_idx))

        self.num_views = len(self.cam_dirs)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.index_map[idx]
        seq_info = self.sequence_data[seq_idx]

        Ks, Ts = seq_info["Ks"], seq_info["Ts"]
        frame_name = seq_info["frame_names"][frame_idx]

        images = []
        for cam_dir in seq_info["cam_dirs"]:
            img_path = os.path.join(cam_dir, frame_name)
            img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)['image']
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img)

        images = torch.stack(images, dim=0)  # (V,C,H,W)

        return {
            "images": images,  # (V,C,H,W)
            "Ks": Ks,
            "Ts": Ts,
            "seq_name": seq_info["name"],
            "frame_idx": frame_idx
        }


def main():
    # ### Setup dataset folder
    root_dir = '/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Images_trampo_avril2025/20250429'
    data_dir = '/home/lea/trampo/MODELS_2D3D/finetuning_multiview/dataset'

    sequences = set([str(f).split('-')[0] for f in os.listdir(root_dir)])
    sequences = sorted([seq for seq in sequences if seq[0] in ['1', '2']])

    cameras = ['Camera1_M11139', 'Camera2_M11140', 'Camera3_M11141', 'Camera4_M11458',
            'Camera5_M11459', 'Camera6_M11461', 'Camera7_M11462', 'Camera8_M11463']

    Ks = np.load('calib/K.npz')['arr_0']
    Ds = np.load('calib/D.npz')['arr_0']

    # Dataloader
    dataset = MultiViewDataset(root_dir=data_dir, K=Ks, transform=None)

    # Set training parameters
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 2
    N = 8 # number of cameras
    K = 17 # number of keypoints

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Init detector
    det_config = '/home/lea/trampo/MODELS_2D3D/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    det_model= init_detector(det_config, det_checkpoint, device=device)
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)

    # Init pose model
    pose_config = '/home/lea/trampo/MODELS_2D3D/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
    pose_model = init_pose_estimator(pose_config, pose_checkpoint, device=device)
    pose_model.train()

    # Unfreeze all parameters
    for param in pose_model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(pose_model.parameters(), lr=LEARNING_RATE)

    # Create a log directory (TensorBoard will read from this)
    writer = SummaryWriter(log_dir="runs/triangulation_experiment")
    print("TensorBoard log dir:", writer.log_dir)

    error_thresh = 50
    person_dist_thresh = 100  # distance threshold to discard mismatched persons
    torch.set_printoptions(precision=4, sci_mode=False)

    Ks = torch.tensor(Ks, dtype=torch.float32, device=device)

    for step, batch in enumerate(dataloader):  # custom dataloader yielding (B,V,C,H,W)
        images, _, Ts, seq, frames = batch.values()
        images, Ts = images.to(device), Ts.to(device)
        
        optimizer.zero_grad()

        # --- 1. Predict 2D keypoints ---
        keypoints = predict_multiview_with_grad(
            det_model, pose_model, images,
            bbox_thr=0.3, pose_batch_size=BATCH_SIZE*N, training=False
        )  # (B,V,K,2)

        # --- 2. Triangulate (batched) ---
        with torch.no_grad():
            error, preds_2d, points_3d = find_best_triangulation(keypoints, Ks, Ts, error_thresh)
            if torch.isnan(points_3d).all():
                continue

            # --- 3. Reproject 3D back into each view ---
            Rt = Ts[:, :, :3, :]
            P_all = Ks @ Rt
            reproj, valid_mask = project_points(points_3d, P_all)

        # --- 4. Compute distances on valid points only ---    
        preds_valid = preds_2d[valid_mask]
        reproj_valid = reproj[valid_mask]

        # --- 5. Remove mismatched persons (dist > 100) ---
        dist = torch.norm(preds_valid - reproj_valid, dim=-1)  # Euclidean distance

        keep_mask = dist < person_dist_thresh
        if keep_mask.sum() == 0:
            print("⚠️ No valid correspondences after distance filtering.")
            continue
        
        preds_valid = preds_valid.detach().clone().requires_grad_(True)
        preds_valid = preds_valid[keep_mask]
        reproj_valid = reproj_valid[keep_mask]

        # --- 6. Compute reprojection loss (RMSE) ---
        loss = torch.sqrt(torch.nn.functional.mse_loss(preds_valid, reproj_valid))
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/reprojection", loss.item(), step)
        writer.add_scalar("Metrics/triangulation_error", torch.mean(error).item(), step)

        #break  # remove once verified

    writer.close()


if __name__ == "__main__":
    main()
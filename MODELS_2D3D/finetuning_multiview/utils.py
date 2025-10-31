import numpy as np
import math
import time
import torch
import cv2
import itertools as it
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/lea/trampo/MODELS_2D3D/mmpose')
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def show_prob(prob_x, prob_y):
    """
    Plot x and y probabilities and mark argmax.

    prob_x: (B, K, W) flatten heatmap x-axis
    prob_y: (B, K, H) flatten heatmap y-axis
    """
    prob_x = prob_x.detach().cpu()
    prob_y = prob_y.detach().cpu()

    B, K, W = prob_x.shape
    B, K, H = prob_y.shape

    fig, axes = plt.subplots(3, 6, figsize=(20, 8))
    axes = axes.flatten()

    b = 0
    x = np.arange(W)
    y = np.arange(H)
    for k in range(K):
        axes[k].plot(x, prob_x[b, k])
        axes[k].scatter(np.argmax(prob_x[b,k]), prob_x[b,k].max() , marker="x", color='red')
        axes[k].set_title(f"K={k}")
        #axes[k].axis("off")

    plt.tight_layout()
    plt.show()

def decode_simcc(x_logits, y_logits, input_size, T=1.0):
    """
    Differentiable decoding of SimCC logits into (x, y) coordinates and confidence.

    Args:
        x_logits: (B, K, Wx)
        y_logits: (B, K, Wy)
        input_size: (H, W)
        T (float): temperature scaling (lower = sharper)
    Returns:
        coords: (B, K, 2)
        conf:   (B, K)
    """
    B, K, Wx = x_logits.shape
    Hy = y_logits.shape[-1]
    H, W = input_size

    # Normalize
    x_logits = x_logits - x_logits.amax(dim=-1, keepdim=True)
    y_logits = y_logits - y_logits.amax(dim=-1, keepdim=True)

    prob_x = F.softmax(x_logits / T, dim=-1)
    prob_y = F.softmax(y_logits / T, dim=-1)

    xs = torch.arange(Wx, device=x_logits.device, dtype=prob_x.dtype)
    ys = torch.arange(Hy, device=y_logits.device, dtype=prob_y.dtype)

    # weighted average instead of argmax
    coord_x = (prob_x * xs).sum(dim=-1)  # (B, K)
    coord_y = (prob_y * ys).sum(dim=-1)  # (B, K)

    # map back to pixel coordinates
    x_img = coord_x / Wx * W  # scale from [0, Wx-1] → [0, W)
    y_img = coord_y / Hy * H  # scale from [0, Hy-1] → [0, H)

    coords = torch.stack([x_img, y_img], dim=-1)  # (B, K, 2)

    # --- Confidence = max prob across both axes ---
    conf_x = prob_x.max(dim=-1)[0]
    conf_y = prob_y.max(dim=-1)[0]
    conf = (conf_x + conf_y) / 2  # average of X/Y confidences

    #print("logits range:", x_logits.min().item(), x_logits.max().item())
    #print("prob max:", prob_x.max().item())
    #print("conf mean:", conf.mean().item())

    return coords, conf

def simcc_to_heatmap(x_logits, y_logits):
    prob_x = F.softmax(x_logits, dim=-1)  # (B, K, Wx)
    prob_y = F.softmax(y_logits, dim=-1)  # (B, K, Hy)
    heatmaps = prob_y[..., :, None] * prob_x[..., None, :]  # (B, K, Hy, Wx)
    return heatmaps

def resize_and_pad_keep_aspect(crop, target_size=(256, 192)):
    """
    Resize crop to target_size while keeping aspect ratio, then pad.
    Args:
        crop: np.ndarray (H, W, C)
        target_size: (W_target, H_target)
    Returns:
        resized_padded: np.ndarray (H_target, W_target, C)
        scale: float (resize factor)
        pad: (pad_left, pad_top)
    """
    H_target, W_target = target_size
    h, w = crop.shape[:2]

    # Compute scale to fit inside target while preserving aspect ratio
    scale = min(W_target / w, H_target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(crop, (new_w, new_h))

    # Compute padding to center the resized image
    pad_x = (W_target - new_w) / 2
    pad_y = (H_target - new_h) / 2

    pad_left = int(np.floor(pad_x))
    pad_right = int(np.ceil(pad_x))
    pad_top = int(np.floor(pad_y))
    pad_bottom = int(np.ceil(pad_y))

    # Pad with zeros (black)
    resized_padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return resized_padded, scale, (pad_left, pad_top)

def map_keypoints_to_bbox(keypoints, scale, pad):
    pad = torch.tensor(pad, dtype=keypoints.dtype, device=keypoints.device)
    keypoints_no_pad = keypoints - pad
    keypoints_orig = keypoints_no_pad / scale
    return keypoints_orig

def show_keypoints_on_crop(image, keypoints):
    if type(image) == np.ndarray:
        im = image
    else:
        im = image.cpu().detach().numpy().squeeze()
        im = np.transpose(im, axes=(1,2,0))
    plt.figure()
    plt.imshow(im)
    keypt = keypoints.cpu().detach().numpy()
    plt.scatter(*keypt.T, s=2, color='orange')
    
    plt.show()

def set_batchnorm_eval(model):
    """Keep BatchNorm layers in eval() while the rest of model stays in train()."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

def predict_multiview_with_grad(
    pose_estimator,
    precomputed_crops,
    precomputed_metas,
    device=None,
    training=False,
    freeze_bn=False,
    T=0.1,
    num_kpts=17):
    """
    Batched multi-view inference so pose_estimator sees many crops at once.
    Only processes views where detections[b,v] == True if detections is given.
    """

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    pose_estimator = pose_estimator.to(device)

    if training:
        pose_estimator.train()
        if freeze_bn:
            set_batchnorm_eval(pose_estimator)
    else:
        pose_estimator.eval()

    model_input_size = (256, 192)
    B, V, N, C, H, W = precomputed_crops.size()
    keypoints_all = torch.full((B, V, N, num_kpts, 3), torch.nan, device=device)

    # Find valid (non-NaN) crops
    valid_mask = ~torch.isnan(precomputed_crops).all(dim=(3, 4, 5))  # (B, V, N)
    valid_indices = valid_mask.nonzero(as_tuple=False)  # list of (b, v, n)

    if len(valid_indices) == 0:
        return keypoints_all  # nothing valid to process

    # Gather valid crops
    valid_crops = precomputed_crops[valid_mask]  # (N_valid, C, H, W)
    batch_tensor = valid_crops.to(device)

    # Handle case where all are NaN (no detections)
    if torch.isnan(batch_tensor).all():
        return keypoints_all
    
    # Forward pass
    with torch.set_grad_enabled(training):
        out_x, out_y = pose_estimator(batch_tensor, None, mode='tensor')
        #out_x_reshaped = out_x.reshape(B, V, N, num_kpts, 3)
        #out_y_reshaped = out_y.reshape(B, V, N, num_kpts, 3)
        keypoints_batch, scores_batch = decode_simcc(out_x, out_y, input_size=model_input_size, T=T)
    
    # Reinsert predictions into full padded tensor
    for i, (b, v, n) in enumerate(valid_indices):
        meta = precomputed_metas[b][v][n]
        if meta is None:
            continue

        kp_crop = keypoints_batch[i]
        sc = scores_batch[i]

        kp_bbox = map_keypoints_to_bbox(kp_crop, meta['scale'], meta['pads'])
        origin = torch.tensor([meta['origin']], dtype=torch.float32, device=device)
        kp_img = kp_bbox + origin

        data = torch.cat([kp_img, sc.unsqueeze(-1)], dim=1)
        keypoints_all[b, v, n] = data

    return keypoints_all


def persons_combinations(keypoints_per_view):
    """
    Generate all possible combinations of detected persons across all views, per batch.

    Args:
        keypoints_per_view: torch.Tensor of shape (B, V, N_local, K, 2 or 3)
            where NaNs indicate missing detections.

    Returns:
        personsIDs_comb: list of length B,
            each element is a (num_combinations_b, V) float tensor of person indices per view.
            Views with no detections are filled with NaN.
    """
    B, V, N_local, K = keypoints_per_view.shape[:4]

    # A detection exists if at least one keypoint is not NaN
    valid = ~torch.isnan(keypoints_per_view[..., 0])  # (B, V, N_local, K)
    nb_persons_per_cam = valid.any(dim=-1).sum(dim=-1)  # (B, V)

    persons_combs_all = []

    for b in range(B):
        nb_persons_per_cam_b = nb_persons_per_cam[b]  # (V,)
        no_detect_mask = nb_persons_per_cam_b == 0

        # Replace 0 by 1 for range() safety
        nb_for_range = torch.where(no_detect_mask, torch.ones_like(nb_persons_per_cam_b), nb_persons_per_cam_b)

        # Create the ranges for each view
        ranges = [range(int(n.item())) for n in nb_for_range]

        # Generate all combinations of person indices
        combs = torch.tensor(list(it.product(*ranges)), dtype=torch.float32)

        # Set NaN where no detection occurred
        if no_detect_mask.any():
            combs[:, no_detect_mask] = float('nan')

        persons_combs_all.append(combs)  # each has shape (num_combinations_b, V)

    return persons_combs_all

def get_persons(keypoints_all, combination, batch_idx):
    """
    Selects the (keypoints, scores) of a specific combination for one batch.
    
    Args:
        keypoints_all: torch.Tensor (B, V, N_local, K, 3)
            Full detections with (x, y, score)
        combination: torch.Tensor (V,)
            Indices of selected detections per view (NaN = no detection)
        batch_idx: int
            Index of the batch to use.
    
    Returns:
        keypoints_selected: (V, K, 2)
        scores_selected: (V, K)
    """
    B, V, N_local, K, D = keypoints_all.shape
    assert D >= 3, "Expected last dim to contain (x, y, score)."

    selected_keypoints = []

    for v in range(V):
        person_idx = combination[v]
        if not math.isnan(person_idx):
            person_idx = int(person_idx.item())
            det = keypoints_all[batch_idx, v, person_idx]  # (K, 3)
            selected_keypoints.append(det)
        else:
            selected_keypoints.append(torch.full((K, 3), float('nan'), device=keypoints_all.device))

    keypoints_selected = torch.stack(selected_keypoints, dim=0)  # (V, K, 3)

    return keypoints_selected

def get_loss(pred, reproj):
    mask = (pred != 0).any(dim=-1) & (reproj != 0).any(dim=-1)  # (N_cams, K)
    mask = mask.unsqueeze(-1)  # (N_cams, K, 1) so it can broadcast with coords

    # Apply mask
    preds_valid = pred[mask.expand_as(pred)].view(-1, 2)
    reproj_valid = reproj[mask.expand_as(reproj)].view(-1, 2)

    loss = torch.sqrt(torch.nn.functional.mse_loss(preds_valid, reproj_valid))
    return loss

def triangulate(points, P):
    """
    Triangulate 3D points from multiple views.
    Args:
        points: torch.Tensor (N_combs, N_cams, K, 2) - 2D points from multiple views
        P: torch.Tensor (N_combs, N_cams, 3, 4) - Projection matrices
    Returns:
        Q: torch.Tensor (N_combs, K, 3) - 3D points
    """
    N_combs, N_cams, K, _ = points.shape
    x_all = points[..., 0]  # (N_combs, N_cams, K)
    y_all = points[..., 1]  # (N_combs, N_cams, K)

    # Expand P to match points shape
    P = P.unsqueeze(2)  # (N_combs, N_cams, 1, 3, 4)
    P_expanded = P.expand(-1, -1, K, -1, -1)  # (N_combs, N_cams, K, 3, 4)

    # Select rows 0,1,2 of P
    P0 = P_expanded[..., 0, :]  # (N_combs, N_cams, K, 4)
    P1 = P_expanded[..., 1, :]
    P2 = P_expanded[..., 2, :]

    # Reshape x, y to match P dimensions
    x = x_all.unsqueeze(-1)  # (N_combs, N_cams, K, 1)
    y = y_all.unsqueeze(-1)  # (N_combs, N_cams, K, 1)

    # Build A matrix rows
    A1 = P0 - x * P2  # (N_combs, N_cams, K, 4)
    A2 = P1 - y * P2
    A = torch.cat([A1, A2], dim=1)  # (N_combs, N_cams*2, K, 4)
    A = A.permute(0, 2, 1, 3).reshape(N_combs * K, N_cams * 2, 4)

    # Batched SVD: returns U, S, Vh with shapes (n_kpts, m, m)
    # We only need Vh
    _, _, Vh = torch.linalg.svd(A)

    # Last column of V (last row of Vh) is solution
    Q_hom = Vh[..., -1, :]  # (n_kpts, 4)
    Q_hom = Q_hom.reshape(N_combs, K, 4)
    Q = Q_hom[..., :3] / Q_hom[..., 3:4]
    return Q

def triangulate_weighted(points, P):
    """
    Triangulate 3D points from multiple views, weighted by keypoint confidences.

    Args:
        points: torch.Tensor (N_combs, N_cams, K, 2)
            2D keypoints from each camera
        P: torch.Tensor (N_combs, N_cams, 3, 4)
            Projection matrices
        confidences: torch.Tensor (N_combs, N_cams, K)
            Confidence/likelihood for each keypoint in each view

    Returns:
        Q: torch.Tensor (N_combs, K, 3)
            Triangulated 3D points
    """
    N_combs, N_cams, K, _ = points.shape
    x_all = points[..., 0]  # (N_combs, N_cams, K)
    y_all = points[..., 1]
    confidences = points[..., 2]  # (N_combs, N_cams, K)

    # Expand P to match points shape
    P_exp = P.unsqueeze(2).expand(-1, -1, K, -1, -1)  # (N_combs, N_cams, K, 3, 4)
    P0, P1, P2 = P_exp[..., 0, :], P_exp[..., 1, :], P_exp[..., 2, :]

    x = x_all.unsqueeze(-1)  # (N_combs, N_cams, K, 1)
    y = y_all.unsqueeze(-1)

    # Core DLT rows
    A1 = P0 - x * P2  # (N_combs, N_cams, K, 4)
    A2 = P1 - y * P2

    # Apply confidence weights like Pose2Sim
    w = confidences.unsqueeze(-1)  # (N_combs, N_cams, K, 1)
    A1 = A1 * w
    A2 = A2 * w

    # Stack and reshape for SVD
    A = torch.cat([A1, A2], dim=1)  # (N_combs, 2*N_cams, K, 4)
    A = A.permute(0, 2, 1, 3).reshape(N_combs * K, 2 * N_cams, 4)

    # Solve via SVD (batched)
    _, _, Vh = torch.linalg.svd(A)
    Q_hom = Vh[..., -1, :]  # (N_combs*K, 4)
    Q_hom = Q_hom.reshape(N_combs, K, 4)

    # Convert from homogeneous coordinates
    Q = Q_hom[..., :3] / Q_hom[..., 3:].clamp(min=1e-8)
    return Q


def triangulate_comb(combs, coords, Ks, Ts):
    """Vectorized version that handles multiple combinations at once"""
    device = combs.device
    batch_size = combs.shape[0]
    
    # Create projection matrices for all cameras
    Rt = Ts[:, :3, :]
    P_all = (Ks @ Rt).squeeze()  # (N_cams, 3, 4)
    
    # Create mask for valid cameras
    valid_mask = ~torch.isnan(combs)  # (batch_size, N_cams)
    valid_counts = valid_mask.sum(dim=1)  # (batch_size,)
    valid_combinations = valid_counts > 1
    
    # Initialize outputs
    Q_combs = torch.full((batch_size, 17, 3), float('nan'), device=device)
    errors = torch.full((batch_size,), float('inf'), device=device)
    
    if not valid_combinations.any():
        return errors, combs, Q_combs
        
    # Handle valid combinations
    coords_expanded = coords.unsqueeze(0)  # (1, N_cams, 17, 2)
    P_expanded = P_all.unsqueeze(0)       # (1, N_cams, 3, 4)
    
    # Expand to batch size
    coords_batch = coords_expanded.expand(batch_size, -1, -1, -1)  # (batch_size, N_cams, 17, 3)
    P_batch = P_expanded.expand(batch_size, -1, -1, -1)           # (batch_size, N_cams, 3, 4)
    
    # Create masked versions where invalid cameras are zeroed out
    mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, N_cams, 1, 1)
    coords_masked = coords_batch * mask_expanded             # (batch_size, N_cams, 17, 3)
    P_masked = P_batch * mask_expanded.expand(-1, -1, 3, 4) # (batch_size, N_cams, 3, 4)
    
    # Triangulate valid combinations
    Q_valid = triangulate(coords_masked[valid_combinations], P_masked[valid_combinations])
    
    # Compute reprojection
    q_calc, _ = project_points(Q_valid, P_masked[valid_combinations])
    
    # Compute errors for valid combinations
    errors_valid = get_loss(coords_masked[valid_combinations, ..., :2], q_calc)
    
    # Assign results back to full tensors
    Q_combs[valid_combinations] = Q_valid
    errors[valid_combinations] = errors_valid
    
    return errors, combs, Q_combs

def find_best_triangulation(keypoints_per_view, Ks, Ts, error_threshold_tracking = 50):
    """
    Find the best matching between detections over all cameras (inspired from personAssociation.py from Pose2Sim)
    Return the triangulation (in Pose2Sim ref frame) of the associated person
    (inspired from best_persons_and_cameras_combination() from Pose2Sim.personAssocation)
    
    Args:
        keypoints: (8, N, 17) list of detections by cameras
        Ks: intrinsics matrices
        Ts: projection matrices

    Returns:
        points_3d: (N, 3) array of 3D points
        confidence_scores: (N,) array of floats

    """
    min_cameras_for_triangulation = 3

    personsIDs_comb = persons_combinations(keypoints_per_view)
    device = keypoints_per_view.device
    B = len(personsIDs_comb)

    ERROR = torch.full((B, 1), torch.nan, dtype=torch.float32, device=device)
    COORDS = torch.full((B, 8, 17, 2), torch.nan, dtype=torch.float32, device=device)
    Q = torch.full((B, 17, 3), torch.nan, dtype=torch.float32, device=device)
    CAMS = torch.full((B, 8), 1, dtype=bool, device=device)
    
    for b, personsIDs_comb_batch in enumerate(personsIDs_comb):
        best_error = float("inf")
        best_Q = None
        best_coords = None

        n_cams = 8
        nb_cams_off = 0
        error_min = float('inf')

        while n_cams - nb_cams_off >= min_cameras_for_triangulation:
            for combination in personsIDs_comb_batch:
                coords = get_persons(keypoints_per_view, combination, b)

                # --- Create subsets with "nb_cams_off" cameras excluded ---
                # Generate all subsets (same device)
                id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off))
                id_cams_off = torch.tensor(id_cams_off, dtype=torch.int64, device=device)  # shape: (n_combos, nb_cams_off)

                # Repeat the combination
                combinations_with_cams_off = combination.repeat(len(id_cams_off), 1).clone()  # (n_combos, n_cams)

                # Build boolean mask (same device)
                mask = torch.zeros_like(combinations_with_cams_off, dtype=torch.bool, device=device)

                # Fill mask at once
                mask.scatter_(1, id_cams_off, True)

                # Apply NaNs
                combinations_with_cams_off = combinations_with_cams_off.to(device)
                combinations_with_cams_off = combinations_with_cams_off.masked_fill(mask, float('nan'))

                # --- Triangulate all subsets at once ---
                error_comb_all, _, Q_comb_all = triangulate_comb(combinations_with_cams_off, coords, Ks[b], Ts[b])

                # --- Evaluate results ---
                error_min = torch.min(error_comb_all)  # error_comb_all is already a tensor
                idx_best = torch.argmin(error_comb_all)

                if error_min < best_error:
                    best_error = error_min
                    best_Q = Q_comb_all[idx_best]
                    best_coords = coords[..., :2]
                    best_cams_off = id_cams_off[idx_best]

            nb_cams_off += 1

        # --- Save best results ---
        if best_error < error_threshold_tracking:
            ERROR[b] = best_error
            COORDS[b] = best_coords
            Q[b] = best_Q
            CAMS[b, best_cams_off] = 0
        
    return ERROR, COORDS, Q, CAMS


def find_triangulation(keypoints_per_view, Ks, Ts, error_threshold_tracking = 50):
    """
    Find the triangulation using the best detections of ALL cameras.
    
    Args:
        keypoints: (8, N, 17) list of detections by cameras
        Ks: intrinsics matrices
        Ts: projection matrices

    Returns:
        points_3d: (N, 3) array of 3D points
        confidence_scores: (N,) array of floats

    """
    personsIDs_comb = persons_combinations(keypoints_per_view)
    device = keypoints_per_view.device
    B = len(personsIDs_comb)

    ERROR = torch.full((B, 1), torch.nan, dtype=torch.float32, device=device)
    COORDS = torch.full((B, 8, 17, 2), torch.nan, dtype=torch.float32, device=device)
    Q = torch.full((B, 17, 3), torch.nan, dtype=torch.float32, device=device)

    for b, personsIDs_comb_batch in enumerate(personsIDs_comb):
        best_error = float("inf")
        best_Q = None
        best_coords = None

        # Try all persons combinations
        for combination in personsIDs_comb_batch:
            #  Get coords
            coords = get_persons(keypoints_per_view, combination, b)
            error_comb, comb, Q_comb = triangulate_comb(combination.unsqueeze(0).to(device), coords, Ks[b], Ts[b])

            if error_comb < best_error:
                best_error = error_comb
                best_Q = Q_comb
                best_coords = coords

        if best_error < error_threshold_tracking:
            ERROR[b] = best_error
            COORDS[b] = best_coords
            Q[b] = best_Q
        #else:
            #print(f'Error is too big: {best_error:.0f} pix.')
    
    return ERROR, COORDS, Q


def project_points(points_3d, P, im_size=(1920, 1080)):
    """
    Projects 3D points to 2D image coordinates using batched camera projection matrices.

    Args:
        points_3d: (B, N, 3) tensor of 3D points.
        P: (B, n_cams, 3, 4) tensor of camera projection matrices.
        im_size: (width, height), optional.
                 If provided, returns a mask for points inside the image.

    Returns:
        points_2d_all: (B, n_cams, N, 2) projected 2D points.
        valid_mask_all: (B, n_cams, N) boolean mask.
    """
    device = points_3d.device
    B, N, _ = points_3d.shape
    _, n_cams, _, _ = P.shape

    # 1. Homogenize 3D points → (B, N, 4)
    ones = torch.ones((B, N, 1), device=device, dtype=points_3d.dtype)
    points_3d_h = torch.cat([points_3d, ones], dim=-1)  # (B, N, 4)

    # 2. Expand to match camera dimension for batched matrix multiply
    #    (B, n_cams, N, 4) → (B, n_cams, 4, N) for matmul
    points_3d_h_exp = points_3d_h[:, None, :, :].expand(-1, n_cams, -1, -1)
    points_3d_h_exp = points_3d_h_exp.permute(0, 1, 3, 2)  # (B, n_cams, 4, N)

    # 3. Project: (B, n_cams, 3, 4) @ (B, n_cams, 4, N) = (B, n_cams, 3, N)
    points_2d_h = torch.matmul(P, points_3d_h_exp)  # (B, n_cams, 3, N)
    points_2d_h = points_2d_h.permute(0, 1, 3, 2)   # (B, n_cams, N, 3)

    # 4. Normalize homogeneous coordinates to get (u, v)
    points_2d_all = points_2d_h[..., :2] / points_2d_h[..., 2:3]  # (B, n_cams, N, 2)

    # 5. Validity mask: points with positive depth and inside image bounds
    z = points_2d_h[..., 2]  # (B, n_cams, N)
    valid_mask_all = z > 0

    if im_size is not None:
        width, height = im_size  # (w, h)
        x, y = points_2d_all[..., 0], points_2d_all[..., 1]
        valid_mask_all &= (x >= 0) & (x < width) & (y >= 0) & (y < height)

    return points_2d_all, valid_mask_all


def show_keypoints_on_im(images, detections, reprojection, savepath, show=False):
    det = detections.cpu().detach().numpy()[0]
    reproj = reprojection.cpu().detach().numpy()[0]

    for v, (im, pt_d, pt_r) in enumerate(zip(images, det, reproj)):
        im = np.transpose(im, axes=(1,2,0))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,6))
        plt.imshow(im)
        plt.scatter(*pt_d.T, s=2, color='orange', label='detection')
        plt.scatter(*pt_r.T, s=2, color='green', label='reprojection')

        plt.legend()
        plt.axis('off')
        
        plt.savefig(f'{savepath}_view{v}.png', bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()
import numpy as np
import math
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

def show_all_keypoints(x_logits, y_logits, input_size, b=0):
    """
    Visualize all keypoints with decoded coordinates overlaid.
    
    x_logits: (B, K, Wx)
    y_logits: (B, K, Hy)
    input_size: (H, W)
    b: batch index to visualize
    """
    coords = decode_simcc(x_logits, y_logits, input_size)  # (B, K, 2)
    heatmaps = simcc_to_heatmap(x_logits, y_logits)        # (B, K, Hy, Wx)

    """ B, K, H, W = heatmaps.shape
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()

    for k in range(K):
        hm = heatmaps[b, k].detach().cpu()
        x, y = coords[b, k].detach().cpu()

        axes[k].imshow(hm, cmap="jet")
        axes[k].scatter([x], [y], c="white", marker="x", s=40)
        axes[k].set_title(f"K={k}")
        #axes[k].axis("off")

    plt.tight_layout()
    plt.show() """

    return coords

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
    detector,
    pose_estimator,
    images,
    bbox_thr=0.3,
    pose_batch_size=8,
    device=None,
    training=False,
    freeze_bn=False,
    T=0.1,
    num_kpts=17
):
    """
    Batched multi-view inference so pose_estimator sees many crops at once.

    Args:
      detector: mmdet detector (frozen)
      pose_estimator: pose model
      images: torch.Tensor (B, V, C, H, W) in [0,1]
      pose_batch_size: number of crops forwarded together
      training: bool, if True enables gradients
      freeze_bn: if True and training==True, BN layers are kept in eval() (good for tiny batches)
      use_softargmax: if True and training==True, uses differentiable soft-argmax (simcc -> coords)
      T: temperature for soft-argmax / softmax

    Returns:
      keypoints_per_view: dict v -> list of (keypoints_img (K,2), scores (K,))
      meta_list (optional): list of meta dicts for each crop (useful for mapping losses)
    """
    device = 'cuda:0' #torch.device(device or ("cpu" if torch.cuda.is_available() else "cpu"))
    pose_estimator = pose_estimator.to(device)

    # Freeze detector (no grads) and put it in eval
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False

    B, V, C, H, W = images.size()
    keypoints_per_view = {v: [] for v in range(V)}

    # to numpy for detector (mmdet expects uint8 BGR/ RGB arrays depending on your setup)
    imgs_np = images.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    imgs_np = imgs_np.astype(np.uint8)

    # collect crops + metadata
    crop_tensors = []
    metas = []  # store mapping so we can place outputs back into keypoints_per_view
    model_input_size = (256, 192)  # (H, W) — make sure this matches your resize_and_pad function
    max_persons = 0

    for b in range(B):
        for v in range(V):
            img = imgs_np[b, v]
            det_result = inference_detector(detector, img)  # still per-image detection
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = pred_instance.bboxes
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > bbox_thr)]
            max_persons = max(max_persons, len(bboxes))

            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox.tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]

                crop_resized, scale, pads = resize_and_pad_keep_aspect(crop, model_input_size)
                # crop_resized is HxWxC (numpy uint8)
                crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0  # (C,H,W)
                crop_tensors.append(crop_tensor)

                origin_bbox = torch.tensor([x1, y1], dtype=torch.float32, device=device)
                pads = torch.tensor(pads, dtype=torch.int16, device=device)
                metas.append({
                    'b': b, 'v': v,
                    'origin': origin_bbox,
                    'scale': scale,
                    'pads': pads
                })

    if len(crop_tensors) == 0:
        return keypoints_per_view, metas  # nothing detected

    # prepare model mode
    if training:
        pose_estimator.train()
        if freeze_bn:
            set_batchnorm_eval(pose_estimator)
    else:
        pose_estimator.eval()

    # --- Prepare tensor for results
    keypoints_all = torch.full(
        (B, V, max_persons, num_kpts, 3),
        torch.nan,
        dtype=torch.float32,
        device=device)

    # run pose model in chunks
    with torch.set_grad_enabled(True):
        batch = torch.stack(crop_tensors, dim=0).to(device)  # (N_batch, C, H, W)

        out_x, out_y = pose_estimator(batch, None, mode='tensor') # out_x: (N_batch, K, W)  out_y: (N_batch, K, H)
        # decode_simcc should return (N_batch, K, 2), (N_batch, K) — adjust if different in your util
        keypoints_batch, scores_batch = decode_simcc(out_x, out_y, input_size=model_input_size, T=T)

        # map back to image coords and store results
        local_indices = {b:{v: 0 for v in range(V)} for b in range(B)}
        for j in range(len(crop_tensors)):  # within this chunk
            meta = metas[j]
            batch_idx = meta['b']
            view_idx = meta['v']
            local_idx = local_indices[batch_idx][view_idx]
            local_indices[batch_idx][view_idx] += 1

            kp_crop = keypoints_batch[j]        # (K,2)
            sc = None if scores_batch is None else scores_batch[j]  # (K,)

            # map_keypoints_to_bbox might expect numpy coords; ensure types match
            kp_bbox = map_keypoints_to_bbox(kp_crop, meta['scale'], meta['pads'])
            origin = meta['origin']
            kp_img = kp_bbox + origin  # (K,2)
            # append to correct view
            data = torch.cat([kp_img, sc.unsqueeze(-1)], dim=1)
            keypoints_all[batch_idx, view_idx, local_idx] = data

    return keypoints_all #, metas, results_per_crop


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
    selected_scores = []

    for v in range(V):
        person_idx = combination[v]
        if not math.isnan(person_idx):
            person_idx = int(person_idx.item())
            det = keypoints_all[batch_idx, v, person_idx]  # (K, 3)
            selected_keypoints.append(det[:, :2])
            selected_scores.append(det[:, 2])
        else:
            selected_keypoints.append(torch.zeros((K, 2), device=keypoints_all.device))
            selected_scores.append(torch.zeros((K,), device=keypoints_all.device))

    keypoints_selected = torch.stack(selected_keypoints, dim=0)  # (V, K, 2)
    scores_selected = torch.stack(selected_scores, dim=0)        # (V, K)

    return keypoints_selected, scores_selected


def euclidean_distance(q1, q2):
    """
    Compute mean per-joint position error (MPJPE) in 2D.

    Args:
        q1, q2: tensors of shape (N_cams, K, 2)

    Returns:
        per_camera: (N_cams,) tensor with mean error per camera
        global_error: scalar with mean error across all cameras and joints
    """
    # difference (N_cams, K, 2)
    dist = q2 - q1

    # mask invalid entries (NaNs)
    valid_mask = ~torch.isnan(dist).any(dim=-1)  # (N_cams, K)

    # squared distances (dx^2 + dy^2)
    dist_sq = (dist ** 2).sum(dim=-1)  # (N_cams, K)

    # replace NaNs with 0 before sqrt
    dist_sq = torch.where(valid_mask, dist_sq, torch.zeros_like(dist_sq))

    # Euclidean distances (N_cams, K)
    euc_dist = torch.sqrt(dist_sq)

    # per-camera MPJPE (mean across keypoints, ignoring NaNs)
    per_camera = torch.zeros(euc_dist.shape[0], device=euc_dist.device)
    for i in range(euc_dist.shape[0]):
        if valid_mask[i].any():
            per_camera[i] = euc_dist[i, valid_mask[i]].mean()
        else:
            per_camera[i] = float("inf")  # no valid keypoints

    # global MPJPE (mean across all cameras + keypoints, ignoring NaNs)
    if valid_mask.any():
        global_error = euc_dist[valid_mask].mean()
    else:
        global_error = torch.tensor(float("inf"), device=euc_dist.device)

    #print(per_camera.size(), torch.min(per_camera), torch.mean(per_camera))
    return per_camera, global_error

def get_loss(pred, reproj):
    mask = (pred != 0).any(dim=-1) & (reproj != 0).any(dim=-1)  # (N_cams, K)
    mask = mask.unsqueeze(-1)  # (N_cams, K, 1) so it can broadcast with coords

    # Apply mask
    preds_valid = pred[mask.expand_as(pred)].view(-1, 2)
    reproj_valid = reproj[mask.expand_as(reproj)].view(-1, 2)

    loss = torch.sqrt(torch.nn.functional.mse_loss(preds_valid, reproj_valid))
    return loss

def triangulate(points, P):

    x_all = points[:,:,0] #.reshape((-1))
    y_all = points[:,:,1] #.reshape((-1))
    n_cams, n_kpts = x_all.shape

    # Build A matrix per camera per keypoint
    # P[:, None] -> (n_cams, 1, 3, 4)
    # x_all.T -> (n_kpts, n_cams)
    P = P.unsqueeze(1)
    P_expanded = P.expand(-1, n_kpts, 3, 4)  # (n_cams, n_kpts, 3, 4)

    # Select rows 0,1,2 of P
    P0 = P_expanded[..., 0, :]  # (n_cams, n_kpts, 4)
    P1 = P_expanded[..., 1, :]
    P2 = P_expanded[..., 2, :]

    x = x_all.unsqueeze(-1)  # (n_kpts, n_cams, 1)
    y = y_all.unsqueeze(-1)

    # Build A rows: (n_cams, n_kpts, 4)
    A1 = P0 - x * P2
    A2 = P1 - y * P2

    # Stack camera equations along rows: (n_cams*2, n_kpts, 4)
    A = torch.cat([A1, A2], dim=0)

    # Permute to (n_kpts, n_cams*2, 4)
    A = A.permute(1, 0, 2)

    # Batched SVD: returns U, S, Vh with shapes (n_kpts, m, m)
    # We only need Vh
    U, S, Vh = torch.linalg.svd(A)

    # Last column of V (last row of Vh) is solution
    Q_hom = Vh[:, -1, :]  # (n_kpts, 4)

    # Normalize homogeneous coordinates
    Q = Q_hom[:, :3] / Q_hom[:, 3:4]

    return Q

def triangulate_comb(comb, coords, Ks, Ts):
    Rt = Ts[:, :3, :]
    P_all = (Ks @ Rt).squeeze()

    # Filter coords and projection_matrices containing nans
    coords_filt = [coords[i] for i in range(len(comb)) if not torch.isnan(comb[i]).item()]
    projection_matrices_filt = [P_all[i] for i in range(len(comb)) if not torch.isnan(comb[i]).item()]
    N_cams = len(coords_filt)

    if N_cams > 1:
        coords_tensor = torch.stack(coords_filt)
        projection_matrices_tensor = torch.stack(projection_matrices_filt)
        Q_comb = triangulate(coords_tensor, projection_matrices_tensor)
    
    else:
        Q_comb = torch.tensor([float('nan'), float('nan'), float('nan')], device='cuda:0').expand(17, 3)
        return torch.tensor(float("inf"), device=Q_comb.device), comb, Q_comb
    
    # Reprojection
    q_calc, _ = project_points(Q_comb.unsqueeze(0), projection_matrices_tensor.unsqueeze(0))

    #error = torch.sqrt(torch.nn.functional.mse_loss(coords_tensor, q_calc))
    error = get_loss(coords_tensor.unsqueeze(0), q_calc)
 
    return error, comb, Q_comb

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

    for b, personsIDs_comb_batch in enumerate(personsIDs_comb):
        best_error = float("inf")
        best_Q = None
        best_coords = None

        n_cams = 8
        error_min = float('inf') 
        nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold

        while n_cams - nb_cams_off >= min_cameras_for_triangulation:
            # Try all persons combinations
            for combination in personsIDs_comb_batch:
                #  Get coords
                coords, scores = get_persons(keypoints_per_view, combination, b)

                # For each persons combination, create subsets with "nb_cams_off" cameras excluded
                id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off))
                combinations_with_cams_off = torch.from_numpy(np.array([combination]*len(id_cams_off))) #, device='cpu')
                for i, id in enumerate(id_cams_off):
                    combinations_with_cams_off[i,id] = float('nan')

                # Try all subsets
                error_comb_all, comb_all, Q_comb_all = [], [], []
                for comb in combinations_with_cams_off:
                    error_comb, comb, Q_comb = triangulate_comb(comb, coords, Ks[b], Ts[b])
                    error_comb_all.append(error_comb)
                    comb_all.append(comb)
                    Q_comb_all.append(Q_comb)

                error_comb_all = torch.stack(error_comb_all)
                error_min = torch.min(error_comb_all)
                idx_best = torch.argmin(error_comb_all)

                if error_min < best_error:
                    best_error = error_min
                    best_Q = torch.stack([Q_comb_all[idx_best]]).squeeze()
                    best_coords = coords
                
            nb_cams_off += 1

        if best_error < error_threshold_tracking:
            ERROR[b] = best_error
            COORDS[b] = best_coords
            Q[b] = best_Q
        else:
            print(f'Error is too big: {best_error:.0f} pix.')
    
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

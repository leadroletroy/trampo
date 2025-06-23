import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import toml
from tqdm import tqdm
from copy import deepcopy
from contextlib import redirect_stdout

#from Pose2Sim.common import computeP

np.set_printoptions(precision=4, suppress=True)

def all_views_processed(root_path, paths, routine):
    for path in paths:
        if path not in os.listdir(root_path):
            print(f'Skipping {routine}, {path} not available')
            return False
        
    return True

def retrieve_keypts(root_path, paths):
    keypts_cam = {}
    for i, path in enumerate(paths):
        with open(os.path.join(root_path, path), "rb") as f:
            data = pickle.load(f)
            keypts_cam[i] = data

    return keypts_cam

## Find the id with the largest vertical amplitude = athlete ###
def find_athlete(keypts_cam):
    # Parameters
    min_size_threshold = 150        # Minimum bbox size
    max_size_threshold = 500        # Maximum bbox size
    amplitude_threshold = 500       # Overall required movement
    max_jump_per_frame = 400        # Max movement between consecutive frames (L2 distance)
    max_dim_change_ratio = 0.7

    id_per_frame_per_cam = {}

    for cam, data in keypts_cam.items():
        athlete_mids, athlete_sizes = [], []
        other_mids, other_sizes = [], []

        last_mid, last_size = None, None
        valid_track = True
        trajectory = []  # stores (frame, mid, size, id)

        sorted_frames = sorted(data.keys())
        for frame in sorted_frames:
            output = data[frame]
            if not output:
                continue

            candidates = []
            for d in output:
                bbox = d['bbox']
                id = d['id']
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                mid = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)

                # Size check
                if size[0] < min_size_threshold or size[1] < min_size_threshold:
                    continue
                if size[0] > max_size_threshold or size[1] > max_size_threshold:
                    continue

                # Height check (above trampoline) for cameras M11139 and M11459
                if cam == 0 :
                    if mid[0] < 500:
                        continue
                elif cam == 4:
                    if mid[0] > (1920 - 450):
                        continue

                candidates.append((mid, size, id))

            if not candidates:
                continue

            # First frame: initialize from first valid candidate
            if last_mid is None:
                mid, size, selected_id = candidates[0]
                trajectory.append((frame, mid, size, selected_id))
                last_mid, last_size = mid, size
                continue

            # Continue track: find closest consistent bbox
            best_candidate = None
            best_jump = float('inf')
            for mid, size, id in candidates:
                jump = np.linalg.norm(np.array(mid) - np.array(last_mid))
                if jump > max_jump_per_frame:
                    print('jump failed')
                    continue

                if jump < best_jump:
                    best_jump = jump
                best_candidate = (mid, size, id)

            if best_candidate:
                mid, size, selected_id = best_candidate
                trajectory.append((frame, mid, size, selected_id))
                last_mid, last_size = mid, size
            """ else:
                print(f"\nCamera {cam}: breaking at frame {frame} due to discontinuity.")
                valid_track = False
                break """

        # Check amplitude over full trajectory
        if valid_track and len(trajectory) > 1:
            mids = np.array([mid for _, mid, _, _ in trajectory])
            amp = np.max(mids, axis=0) - np.min(mids, axis=0)
            total_amp = np.linalg.norm(amp)
            if total_amp > amplitude_threshold:
                for frame, mid, size, id in trajectory:
                    athlete_mids.append(mid)
                    athlete_sizes.append(size)
                    id_per_frame_per_cam.setdefault(cam, {}).setdefault(frame, []).append(id)
            else:
                #print('total amp failed')
                for _, mid, size, _ in trajectory:
                    other_mids.append(mid)
                    other_sizes.append(size)
        else:
            for _, mid, size, _ in trajectory:
                other_mids.append(mid)
                other_sizes.append(size)

        """ # Print stats
        print(f"Camera: {cam}")
        if athlete_mids:
            mids_np = np.array(athlete_mids)
            sizes_np = np.array(athlete_sizes)
            print("Athlete mid stats:")
            print("  min:", np.min(mids_np, axis=0), "  max:", np.max(mids_np, axis=0), "  mean:", np.mean(mids_np, axis=0))
            print("Athlete bbox size stats:")
            print("  min:", np.min(sizes_np, axis=0), "  max:", np.max(sizes_np, axis=0), "  mean:", np.mean(sizes_np, axis=0))

        if other_mids:
            mids_np = np.array(other_mids)
            sizes_np = np.array(other_sizes)
            print("Other mid stats:")
            print("  min:", np.min(mids_np, axis=0), "  max:", np.max(mids_np, axis=0), "  mean:", np.mean(mids_np, axis=0))
            print("Other bbox size stats:")
            print("  min:", np.min(sizes_np, axis=0), "  max:", np.max(sizes_np, axis=0), "  mean:", np.mean(sizes_np, axis=0)) """

    return id_per_frame_per_cam

def select_keypoints_athlete(keypts_cam, id_per_frame_per_cam):
    keypts_per_cam = {}
    for cam, data in keypts_cam.items():
        keypts = []
        for frame, output in data.items():
            for d in output:
                id = d['id']
                if cam in id_per_frame_per_cam and frame in id_per_frame_per_cam[cam] and id in id_per_frame_per_cam[cam][frame]: # == id:
                    pts = [p[0:2] for p in d['keypoints']]
                    conf = [p[2] for p in d['keypoints']]
                    keypts.append((frame, pts, conf))

        keypts_per_cam[cam] = keypts

    return keypts_per_cam

# TODO: ajouter filtre pour associer ids correspondants et conserver une seule personne SMPL

def retrieve_common_id(keypts_cam):
    """
    Find the most seen id in all the sequence and set it as the athlete id.
    If only one person present in a frame, also set as the athlete (to confirm).
    """
    id_per_frame_per_cam = {}

    for cam, data in keypts_cam.items():
        id_count = {i:0 for i in range(-1, 20)}

        sorted_frames = sorted(data.keys())
        for frame in sorted_frames:
            output = data[frame]

            #print(output)

            if not output:
                continue

            elif len(output) == 1:
                id = output[0]['id']
                id_count[id] += 1
                id_per_frame_per_cam.setdefault(cam, {}).setdefault(frame, []).append(id)

            else:
                for d in output:
                    id = d['id']
                    id_count[id] += 1
        
        most_seen_id = sorted(id_count.items(), key=lambda x: x[-1])[-1][0]

        for frame in sorted_frames:
            if frame not in id_per_frame_per_cam[cam]:
                id_per_frame_per_cam.setdefault(cam, {}).setdefault(frame, []).append(most_seen_id)

    return id_per_frame_per_cam

def vis_keypoints_mvt(keypts_per_cam, cam):
    keypts = keypts_per_cam[cam]

    # Initialize lists for each coordinate per keypoint
    num_keypoints = 45
    frame_vals = [[] for _ in range(num_keypoints)]
    x_vals = [[] for _ in range(num_keypoints)]
    y_vals = [[] for _ in range(num_keypoints)]
    score_vals = [[] for _ in range(num_keypoints)]

    # Collect values for person id == 0
    for frame, pts, score in keypts:
        for i in range(num_keypoints):
            frame_vals[i].append(int(frame.split('_')[1].split('.')[0]))
            x_vals[i].append(pts[i][0])
            y_vals[i].append(pts[i][1])
            score_vals[i].append(score[i])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10,5), sharex=True)
    axes[0].set_title(f'Keypoints in Camera {cam+1} view during exercise')
    coords = ['x', 'y', 'score']
    all_vals = [frame_vals, x_vals, y_vals, score_vals]

    for i, ax in enumerate(axes):
        for kpt_id in range(num_keypoints):
            ax.plot(all_vals[0][kpt_id], all_vals[i+1][kpt_id], label=f'kpt_{kpt_id+1}')
        ax.set_ylabel(coords[i])
        ax.legend(loc='upper right', ncol=2, fontsize='small')
        ax.grid(True)

    axes[-1].set_xlabel('Frame index')
    plt.tight_layout()
    plt.show()
    return

def find_frames_detections_all_views(keypts_per_cam):
    lists = list(keypts_per_cam.values())
    key_sets = [set(row[0] for row in l) for l in lists]
    common_keys = set.intersection(*key_sets)

    filtered_lists = [[row for row in l if row[0] in common_keys] for l in lists]
    print(f'{len(filtered_lists[0])} frames with detections in all views')

    return filtered_lists

def create_dict_all_detections(keypts_cam, N_images, cameras, id_per_frame_per_cam, input_data):
    keypts_per_cam_all = {}
    for cam, data in keypts_cam.items():
        keypts = []
        for frame, output in data.items():
            for d in output:
                id = d['id']
                if cam in id_per_frame_per_cam and frame in id_per_frame_per_cam[cam]: # == id:

                    if input_data == '2d':
                        pts = [p[0:2] for p in d['keypoints']]
                        conf = [p[2] for p in d['keypoints']]
                        keypts.append((frame, pts, conf))

                    elif input_data == '3d':
                        pts = [p[0:3] for p in d['keypoints']]
                        keypts.append((frame, pts, 0))

        keypts_per_cam_all[cam] = keypts

    keypoints_detections = {}

    for frame_nb in range(N_images):
        if frame_nb not in keypoints_detections:
            keypoints_detections.update({frame_nb:{}})
        for c, cam in enumerate(cameras):
            for el in keypts_per_cam_all[c]:
                if f'{frame_nb:05d}' in el[0]:
                    if cam not in keypoints_detections[frame_nb]:
                        keypoints_detections[frame_nb].update({cam:[el[1]]})
                    else:
                        keypoints_detections[frame_nb][cam].append(el[1])
    
    return keypoints_detections

def remove_outliers(data, multiplier=2, max_threshold=1000):
    """
    Supprime les outliers d'un tableau 1D selon la règle de 1.5*IQR
    et retourne les données filtrées ainsi que les outliers détectés.
    """
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    filtered = data[(data >= lower) & (data <= upper) & (data <= max_threshold)]
    outliers = data[(data < lower) | (data > upper) | (data >= max_threshold)]
    return filtered, outliers

def computeP(calib_file, undistort=False):
    '''
    Compute projection matrices from toml calibration file. (from Pose2Sim)
    
    INPUT:
    - calib_file: calibration .toml file.
    - undistort: boolean
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    calib = toml.load(calib_file)
    
    cal_keys = [c for c in calib.keys() 
                if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
                and isinstance(calib[c], dict)]
    P = []
    K_all = []
    for cam in list(cal_keys):
        K = np.array(calib[cam]['matrix'])
        if undistort:
            S = np.array(calib[cam]['size'])
            dist = np.array(calib[cam]['distortions'])
            optim_K = cv2.getOptimalNewCameraMatrix(K, dist, [int(s) for s in S], 1, [int(s) for s in S])[0]
            Kh = np.block([optim_K, np.zeros(3).reshape(3,1)])
        else:
            Kh = np.block([K, np.zeros(3).reshape(3,1)])
        R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
        T = np.array(calib[cam]['translation'])
        H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
        
        P.append(Kh @ H)
        K_all.append(Kh)
   
    return P, K_all

def triangulate_reproject(keypoints_detections, cameras, N_images):
    error_cam_1 = {f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras}
    error_cam_2 = deepcopy(error_cam_1)

    error_cam = [{f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras} for _ in range(8)]

    triangulated_points = {f'{c1}-{c2}':[] for c1 in range(8) for c2 in range(8) if c1 < c2}

    P_all, K_all = computeP('/home/lea/trampo/Pose2Sim/Calibration/Calib.toml')
    #print(P_all)

    for c1 in range(8):
        for c2 in range(c1, 8):
            if c1 != c2:
                for frame_idx in range(N_images):
                    try:
                        pts1 = np.asarray(keypoints_detections[frame_idx][cameras[c1]])[0, :, :]
                        pts2 = np.asarray(keypoints_detections[frame_idx][cameras[c2]])[0, :, :]
                    except KeyError:
                        continue

                    #TODO: ajouter filtre pour vérifier qu'il s'agit de la même personne (en remplacement de [0, :, :] ci-haut)
                    
                    P1 = P_all[c1]
                    P2 = P_all[c2]

                    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                    points_3d = (points_4d[:3] / points_4d[3]).T  # Nx3
                    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T  # 4xN

                    triangulated_points[f'{c1}-{c2}'].append(points_3d)

                    pts1_reproj_h = P1 @ points_3d_h  # 3xN
                    pts1_reproj = (pts1_reproj_h[:2] / pts1_reproj_h[2]).T  # Nx2

                    pts2_reproj_h = P2 @ points_3d_h  # 3xN
                    pts2_reproj = (pts2_reproj_h[:2] / pts2_reproj_h[2]).T  # Nx2

                    error1 = np.linalg.norm(pts1 - pts1_reproj, axis=1)
                    error2 = np.linalg.norm(pts2 - pts2_reproj, axis=1)
                    error_cam_1[f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error1))
                    error_cam_2[f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error2))

                    for c in range(8):
                        try:
                            pts = np.asarray(keypoints_detections[frame_idx][cameras[c]])[0, :, :]
                        except KeyError:
                            continue
                        pts_reproj_h = P_all[c] @ points_3d_h
                        pts_reproj = (pts_reproj_h[:2] / pts_reproj_h[2]).T 
                        error = np.linalg.norm(pts - pts_reproj, axis=1)

                        error_cam[c][f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error))

    error_cam_1_filtered = {k: v for k, v in error_cam_1.items() if len(v) > 0}
    error_cam_2_filtered = {k: v for k, v in error_cam_2.items() if len(v) > 0}

    return error_cam_1_filtered, error_cam_2_filtered, error_cam, triangulated_points

def reproject(keypoints_detections, cameras, N_images):
    error_cam_1 = {f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras}
    error_cam_2 = deepcopy(error_cam_1)

    error_cam = [{f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras} for _ in range(8)]

    P_all = computeP('/home/lea/trampo/Pose2Sim/Calibration/Calib.toml')

    T = np.load('/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Calib_trampo_avril2025/results_calib/calib_0429/WorldTCam_opt.npz')['arr_0']

    for c1 in range(8):
        T1 = T[c1]
        for c2 in range(c1, 8):
            T2 = T[c2]
            if c1 != c2:
                for frame_idx in range(N_images):

                    try:
                        pts3d_1 = np.asarray(keypoints_detections[frame_idx][cameras[c1]])[0, :, :]
                        pts3d_2 = np.asarray(keypoints_detections[frame_idx][cameras[c2]])[0, :, :]
                    except KeyError:
                        continue

                    #TODO: ajouter filtre pour vérifier qu'il s'agit de la même personne (en remplacement de [0, :, :] ci-haut)
                    
                    pts3d_1 *= 1000 # convert to mm
                    pts3d_2 *= 1000 # convert to mm

                    pts3d_1_h = np.hstack((pts3d_1, np.ones((pts3d_1.shape[0], 1)))).T  # 4xN
                    pts3d_2_h = np.hstack((pts3d_2, np.ones((pts3d_2.shape[0], 1)))).T  # 4xN

                    #print('Original points')
                    #print(pts3d_1)

                    # FORMULA: 1T2 = inv(T1) @ T2

                    pts4d_1_2 = np.linalg.inv(T2) @ T1 @ pts3d_1_h # 4xN
                    pts4d_2_1 = np.linalg.inv(T1) @ T2 @ pts3d_2_h # 4xN

                    pts3d_1_2 = pts4d_1_2.T[:, :3] / pts4d_1_2.T[:, 3].reshape(-1, 1) # Nx3
                    pts3d_2_1 = pts4d_2_1.T[:, :3] / pts4d_2_1.T[:, 3].reshape(-1, 1) # Nx3

                    #print('Reprojected points')
                    #print(pts3d_2_1)

                    error1 = np.linalg.norm(pts3d_1 - pts3d_2_1, axis=1)
                    error2 = np.linalg.norm(pts3d_2 - pts3d_1_2, axis=1)
                    error_cam_1[f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error1))
                    error_cam_2[f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error2))

                    """ for c in range(8):
                        try:
                            pts = np.asarray(keypoints_detections[frame_idx][cameras[c]])[0, :, :]
                        except KeyError:
                            continue

                        pts_reproj_h = P_all[c] @ points_3d_h
                        pts_reproj = (pts_reproj_h[:2] / pts_reproj_h[2]).T 
                        error = np.linalg.norm(pts - pts_reproj, axis=1)

                        error_cam[c][f'{cameras[c1][-2:]}-{cameras[c2][-2:]}'].append(np.mean(error)) """

    error_cam_1_filtered = {k: v for k, v in error_cam_1.items() if len(v) > 0}
    error_cam_2_filtered = {k: v for k, v in error_cam_2.items() if len(v) > 0}

    return error_cam_1_filtered, error_cam_2_filtered, error_cam

def compute_mpjpe_variance(keypoints_list):
    """
    Computes the average MPJPE-like variance from a list of 3D keypoints sets.
    
    Parameters:
        keypoints_list (List[np.ndarray]): list of arrays of shape (num_joints, 3)
    
    Returns:
        mean_joint_variance (float): average distance of each joint from the mean joint
        jointwise_variances (np.ndarray): per-joint variances (shape: num_joints,)
    """
    keypoints_array = np.stack(keypoints_list)  # shape: (N, J, 3)
    mean_pose = np.mean(keypoints_array, axis=0)  # shape: (J, 3)
    
    # Euclidean distance from mean pose, shape: (N, J)
    errors = np.linalg.norm(keypoints_array - mean_pose, axis=2)
    N = errors.shape[0]
    
    # Mean over all frames and joints
    mean_joint_variance = np.mean(errors)
    min_joint_variance = np.min(errors)
    max_joint_variance = np.max(errors)
    
    # Mean variance per joint
    jointwise_variances = np.mean(errors, axis=0)  # shape: (J,)
    
    return N, mean_joint_variance, jointwise_variances, min_joint_variance, max_joint_variance

def save_mpjpe_variance(triangulated_points):
    with open("variance_log.txt", "w") as f:
        with redirect_stdout(f):
            for c, keypoints_list in triangulated_points.items():
                print(f'{cameras[int(c[0])]} - {cameras[int(c[-1])]}')

                N, mean_error, per_joint_error, min_error, max_error = compute_mpjpe_variance(keypoints_list)

                print(f'N: {N}')
                print(f'Mean joint variance: {mean_error:.2f} mm')
                print(f'Min joint variance: {min_error:.2f} mm')
                print(f'Max joint variance: {max_error:.2f} mm')
                print(f'Per-joint variance: {per_joint_error} \n')
    return

def plot_view_consistency(error_cam, no_outliers=True):

    error_cam = {k: error_cam[k] for k in sorted(error_cam)}

    labels = []
    data_filtered = []
    means = []
    stds = []

    print("Outliers par paire de caméras :")
    for key, val in error_cam.items():
        if no_outliers:
            val_filtered, outliers = remove_outliers(val, 2, np.inf)
            if len(val_filtered) == 0:
                print('Only outliers, skip')
                continue  # skip if all values are outliers
        else:
            val_filtered = val
            outliers = []
            
        labels.append(key)
        data_filtered.append(val_filtered)
        mean_val = np.mean(val_filtered)
        std_val = np.std(val_filtered)
        means.append(mean_val)
        stds.append(std_val)
        print(f"{key}: {len(val_filtered)} accepted, mean = {np.mean(val_filtered)}")
        if len(outliers) > 0:
            print(f"{key}: {len(outliers)} outlier(s), mean = {np.mean(outliers)}")
        
    # Création du boxplot sans outliers
    plt.figure(figsize=(14,7))

    box = plt.boxplot(data_filtered, labels=labels, patch_artist=True,
                      showfliers=False, medianprops={'color': 'red'})

    # Couleurs personnalisées (facultatif)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_filtered)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Ajouter les moyennes et écarts-types
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i + 1, mean, f"μ = {mean:.0f}\nσ = {std:.0f}", 
                ha='center', fontsize=8, color='green',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.7))

    # Mise en page
    avec_sans_out = 'sans' if no_outliers else 'avec'
    plt.title(f"Erreur de reprojection par paire de caméras ({avec_sans_out} outliers)")
    plt.xlabel("Paire de caméras")
    plt.ylabel("Erreur de reprojection (pixels)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'Reproj_all_{avec_sans_out}_outliers')
    plt.show()
    
    return

def plot_consistency_per_view(error_cam, cameras):
    for v, error_cam_i in enumerate(error_cam):
        error_cam_i = {k: error_cam_i[k] for k in sorted(error_cam_i)}
        error_cam_i = {k: v for k, v in error_cam_i.items() if len(v) > 0}

        labels = []
        data_filtered = []
        means = []
        stds = []
        used_images = 0

        #print("Outliers par paire de caméras :")
        for key, val in error_cam_i.items():
            val_filtered, outliers = remove_outliers(val)
            if len(val_filtered) == 0:
                continue  # skip if all values are outliers
            
            used_images += len(val_filtered)
            labels.append(key)
            data_filtered.append(val_filtered)
            mean_val = np.mean(val_filtered)
            std_val = np.std(val_filtered)
            means.append(mean_val)
            stds.append(std_val)

        # Création du boxplot sans outliers
        plt.figure(figsize=(14,7))
        box = plt.boxplot(data_filtered, labels=labels, patch_artist=True,
                            showfliers=False, medianprops={'color': 'red'})

        # Couleurs personnalisées (facultatif)
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_filtered)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Ajouter les moyennes et écarts-types
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i + 1, mean, f"μ = {mean:.0f}\nσ = {std:.0f}", 
                    ha='center', fontsize=8, color='green',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.7))

        # Mise en page
        plt.title(f"Erreur de reprojection sur la vue {v+1} ({cameras[v]}) par paire de caméras (sans outliers, {used_images} projections)")
        plt.xlabel("Paire de caméras")
        plt.ylabel("Erreur de reprojection (pixels)")
        plt.ylim(0, 500)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'Reproj_vue{v+1}')
        plt.show()
    return

def save_error(error, filename):
    with open(filename, 'w') as f:
        json.dump(error, f)
    return

def save_points(points, filename):
    with open(filename, 'wb') as f:
        pickle.dump(points, f)
    return

#TODO: ajouter visulisation des points 3D

if __name__ == "__main__":

    input_data = sys.argv[-1]
    print('ARGV:', sys.argv, input_data, '\n')

    video_path = '/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Images_trampo_avril2025/20250429/'
    root_path = '/home/lea/trampo/metrics/SMPL_keypoints'

    cameras = ['M11139', 'M11140', 'M11141', 'M11458', 'M11459', 'M11461', 'M11462', 'M11463']
    routines = sorted([os.path.basename(f).split('-')[0] for f in os.listdir(root_path)])

    error_cam_all = {}
    triangulated_points_all = {f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras}
    error_cam_per_view_all = [{f'{c1[-2:]}-{c2[-2:]}':[] for c1 in cameras for c2 in cameras} for _ in range(8)]

    for routine in tqdm(routines):
        paths = [routine+'-Camera1_M11139_2d.pkl', routine+'-Camera2_M11140_2d.pkl', routine+'-Camera3_M11141_2d.pkl', routine+'-Camera4_M11458_2d.pkl',
                 routine+'-Camera5_M11459_2d.pkl', routine+'-Camera6_M11461_2d.pkl', routine+'-Camera7_M11462_2d.pkl', routine+'-Camera8_M11463_2d.pkl',]

        if all_views_processed(root_path, paths, routine):
            N_images = len(os.listdir(video_path+routine+'-Camera1_M11139'))
            keypts_cam = retrieve_keypts(root_path, paths)

            if input_data == '2d':
                print('2d data processing')
                #id_per_frame_per_cam = find_athlete(keypts_cam)
                id_per_frame_per_cam = retrieve_common_id(keypts_cam)
                keypts_per_cam = select_keypoints_athlete(keypts_cam, id_per_frame_per_cam)
                keypoints_detections = create_dict_all_detections(keypts_cam, N_images, cameras, id_per_frame_per_cam, input_data)

                error_cam_1, error_cam_2, error_cam_per_view, triangulated_points = triangulate_reproject(keypoints_detections, cameras, N_images)
                
            elif input_data == '3d':
                print('3d data processing')
                id_per_frame_per_cam = retrieve_common_id(keypts_cam)
                keypts_per_cam = select_keypoints_athlete(keypts_cam, id_per_frame_per_cam)
                keypoints_detections = create_dict_all_detections(keypts_cam, N_images, cameras, id_per_frame_per_cam, input_data)

                error_cam_1, error_cam_2, error_cam_per_view = reproject(keypoints_detections, cameras, N_images)

            all_keys = set(error_cam_1) | set(error_cam_2) | set(error_cam_all)

            error_cam_all = {k: error_cam_1.get(k, []) + error_cam_2.get(k, []) + error_cam_all.get(k, []) for k in all_keys}

            for i, el in enumerate(error_cam_per_view):
                error_cam_per_view_all[i] = {k: error_cam_per_view_all[i].get(k, []) + el.get(k, []) for k in all_keys}
            
            for key, el in triangulated_points.items():
                triangulated_points_all = {key: triangulated_points_all.get(key, []) + el}

    print(triangulated_points.keys())

    save_error(error_cam_all, 'error_4dhumans_filt.json')
    save_error(error_cam_per_view, 'error_4dhumans_perview_filt.json')
    save_points(triangulated_points_all, 'triangulated_points_filt.pkl')
    plot_view_consistency(error_cam_all, no_outliers=False)
    plot_consistency_per_view(error_cam_per_view_all, cameras)

    save_mpjpe_variance(triangulated_points_all)


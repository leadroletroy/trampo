import os
import cv2
import numpy as np
import pandas as pd
import re
import joblib
import json

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as rot
from scipy.spatial import procrustes

from tqdm import tqdm

# --- Load and parse .trc file ---
def load_trc(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # --- Metadata ---
    data_start_line = 5
    header_line = lines[3]
    header_parts = re.split(r'\s+', header_line.strip())
    
    marker_names = header_parts[2:]  # skip "Frame#" and "Time"
    n_markers = len(marker_names)
    
    # Reconstruct columns: Frame#, Time, then X/Y/Z for each marker
    cols = ['Frame#', 'Time']
    for name in marker_names:
        cols.append(f'{name}_X')
        cols.append(f'{name}_Y')
        cols.append(f'{name}_Z')

    # Read numerical data starting from line 6
    df = pd.read_csv(filename, sep=r'\s+|\t+', engine='python', skiprows=data_start_line, header=None)
    df.columns = cols

    return df, marker_names

def extract_trc_positions(df, marker_names):
    return np.stack([
        df[[f"{name}_X", f"{name}_Y", f"{name}_Z"]].values
        for name in marker_names
    ], axis=1)  # shape: (n_frames, n_markers, 3)

def get_trc_frames(df, marker_names):
    # List of all X/Y/Z coordinate columns
    coord_cols = [f"{name}_{axis}" for name in marker_names for axis in ['X', 'Y', 'Z']]

    # Boolean mask: True if at least one coordinate in the row is not NaN
    detection_mask = df[coord_cols].notna().any(axis=1)

    # Extract corresponding frame numbers
    detected_frames = df.loc[detection_mask, 'Frame#'].astype(int).tolist()

    return detected_frames

def project(joints_3d, cam_trans, size):
    f = 5000

    K = np.array([[f/256, 0, 0],
                [0, f/256, 0],
                [0, 0, 1]])
    R = np.eye(3)

    points = joints_3d + cam_trans # (4,3)

    proj = (K @ R @ points.T).T # (4,3)
    proj_norm = proj[:,:2] / proj[:,2].reshape(-1,1)
    proj_im = proj_norm * max(size) * scale
    proj_im += np.array(size[::-1])/2
    return proj_im

def load_3d_keypoints(filename, transfo=True):
    results = joblib.load(filename)
    all_joints = []

    for frame_idx in sorted(results, key=lambda x: x[-9:]):
        frame_res = results[frame_idx]

        # If there's only one person
        if len(frame_res['3d_joints']) > 0:
            joints_3d = frame_res['3d_joints']  # shape (N, 3)
            
            if transfo:
                transformed_joints = []
                # Apply transformations
                for i in range(len(joints_3d)):
                    cam_trans = frame_res['pose'][i][-3:].reshape((1,3))
                    orient = frame_res['smpl'][i]['global_orient']
                    scale = frame_res['scale'][i]

                    T_cam = np.eye(4)
                    T_cam[0:3, 0:3] = orient

                    pts_3d = joints_3d[i] #* scale

                    pts_trans = pts_3d - cam_trans
                    pts_trans_rot = np.linalg.inv(T_cam) @ np.vstack((pts_trans.T, np.ones((1, 45))))
                    pts_trans_rot = pts_trans_rot.T[:,0:3]
                    transformed_joints.append(pts_trans_rot)

                all_joints.append(transformed_joints) # shape (d, N, 3)
            else:
                all_joints.append(joints_3d)
        else:
            all_joints.append([np.full((45,3), np.nan)])  # or zero if missing

    num_frames = len(all_joints)
    max_persons = max(len(frame_list) for frame_list in all_joints)
    num_joints = all_joints[0][0].shape[0]  # Assuming all persons have same joints count

    for t in range(num_frames):
        persons_in_frame = len(all_joints[t])
        if persons_in_frame < max_persons:
            # create padding arrays full of nan
            pad_count = max_persons - persons_in_frame
            nan_pad = [np.full((num_joints, 3), np.nan)] * pad_count
            all_joints[t].extend(nan_pad)

    # Now convert to np.array
    all_joints_array = np.array(all_joints)

    return all_joints_array  # shape (n_frames, n_joints, 3)

def procrustes_align(A_pts, B_pts):
    """
    Computes optimal rotation, translation, and scale to align B_pts to A_pts.
    Returns: B_aligned (N, 3), R (3,3), scale (float), t (1,3)
    """
    A_mean = A_pts.mean(axis=0)
    B_mean = B_pts.mean(axis=0)

    A_c = A_pts - A_mean
    B_c = B_pts - B_mean

    norm_A = np.linalg.norm(A_c)
    norm_B = np.linalg.norm(B_c)

    A_c /= norm_A
    B_c /= norm_B

    H = B_c.T @ A_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = (S.sum()) * (norm_A / norm_B)
    t = A_mean - (B_mean @ R) * scale

    return R, scale, t

def kabsch_align(A_pts, B_pts):
    """
    Computes optimal rotation and translation to align B_pts to A_pts (no scaling).
    Returns: R (3,3), t (1,3)
    """
    A_mean = A_pts.mean(axis=0)
    B_mean = B_pts.mean(axis=0)

    A_c = A_pts - A_mean
    B_c = B_pts - B_mean

    H = B_c.T @ A_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = A_mean - B_mean @ R

    return R, t

def draw_skeleton(points_2d, color, ax, model='vit'):
    if model == '4dhumans':
        skeleton = [
            (1, 8),    # Neck -> Mid Hip
            (1, 5),    # Neck -> R Shoulder
            (5, 6),    # R Shoulder -> R Elbow
            (6, 7),    # R Elbow -> R Wrist
            (1, 2),    # Neck -> L Shoulder
            (2, 3),    # L Shoulder -> L Elbow
            (3, 4),    # L Elbow -> L Wrist
            (8, 12),    # Mid Hip -> R Hip
            (12, 13),   # R Hip -> R Knee
            (13, 14),  # R Knee -> R Ankle
            (8, 9),   # Mid Hip -> L Hip
            (9, 10),  # L Hip -> L Knee
            (10, 11),  # L Knee -> L Ankle
            (0, 1),    # Neck -> Nose
            ]
    elif model == 'vit':
        skeleton = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (3, 5),
            (2, 4),
            (4, 6),
            (1, 7),
            (2, 8),
            (7, 8),
            (7, 9),
            (9, 11),
            (8, 10),
            (10, 12)
            ]
    
    for i, j in skeleton:
        xline = [points_2d[i][0], points_2d[j][0]]
        yline = [points_2d[i][1], points_2d[j][1]]
        if points_2d[i].shape[-1] == 3:
            zline = [points_2d[i][2], points_2d[j][2]]
            ax.plot(xline, yline, zline, color=color)
        else:
            ax.plot(xline, yline, color=color)
    return

### Visualize joints alignment across time
def plot_joint_coordinates_over_time(A, B, C, D, joint_names=None, title_prefix=''):
    
    #A: (F, 13, 3) - ground truth joints over frames
    #B_selected: (F, 13, 3) - matched joints from B
    #joint_names: list of joint names (optional)
    
    F, K, _ = A.shape
    t = np.arange(F)

    # Compute global y-limits per axis (X, Y, Z)
    y_mins = [np.min([A[:, :, i], B[:, :, i], C[:, :, i], D[:, :, i]]) for i in range(3)]
    y_maxs = [np.max([A[:, :, i], B[:, :, i], C[:, :, i], D[:, :, i]]) for i in range(3)]

    # Compute global y-limits (all X, Y, Z)
    all_data = np.concatenate([
        A.reshape(-1),
        B.reshape(-1),
        C.reshape(-1),
        D.reshape(-1)
    ])
    y_min, y_max = all_data.min(), all_data.max()


    # Set up grid: K rows (joints), 3 columns (X, Y, Z)
    fig, axes = plt.subplots(K, 3, figsize=(15, 3 * K), sharex=True)
    if K == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D even when K == 1

    for j in range(K):
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax = axes[j, i]
            ax.plot(t, A[:, j, i], label='A', color='blue')
            ax.plot(t, B[:, j, i], label='B', color='orange', alpha=0.7)
            ax.plot(t, C[:, j, i], label='B global', color='green', alpha=0.7)
            ax.plot(t, D[:, j, i], label='B KA', color='red', alpha=0.7)
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{axis} (m)')
            title = f'{axis}-coord, Joint {j}'
            if joint_names:
                title += f" ({joint_names[j]})"
            ax.set_title(title)

            # Set common y-limits per axis column
            #ax.set_ylim(y_mins[i], y_maxs[i])
            ax.set_ylim(y_min, y_max)

            if i == 0:
                ax.legend(loc='upper right')

    fig.suptitle(title_prefix.strip(), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'metrics/3dregistrationFigs/Joints_{routine}_{camera}.png')
    #plt.show()


if __name__ == '__main__':
    cameras = ['Camera1_M11139', 'Camera2_M11140', 'Camera3_M11141', 'Camera4_M11458', 'Camera5_M11459', 'Camera6_M11461', 'Camera7_M11462', 'Camera8_M11463']
    routines = [f.split('.')[0] for f in sorted(os.listdir('/home/lea/trampo/Pose2Sim/pose-3d-4DHumans'))]

    pose2sim = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel',
                'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist']
    humans = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
            'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'Rear', 'LEar']

    common_indices = [j for j in humans if j in pose2sim]

    # Get the index order to match Pose2Sim's triangulated keypoints
    matching_pose2sim = [pose2sim.index(j) for j in common_indices]
    matching_humans = [humans.index(j) for j in common_indices]

    mpjpe_raw_dict = {}
    mpjpe_global_dict = {}
    mpjpe_ka_dict = {}
    mpjpe_pa_dict = {}

    for routine in tqdm(routines):
        mpjpe_raw_dict.update({routine: {c:[] for c in cameras}})
        mpjpe_global_dict.update({routine: {c:[] for c in cameras}})
        mpjpe_ka_dict.update({routine: {c:[] for c in cameras}})
        mpjpe_pa_dict.update({routine: {c:[] for c in cameras}})

        for cam_idx in range(8):

            camera = cameras[cam_idx]

            trc_path = f"/home/lea/trampo/Pose2Sim/pose-3d-4DHumans/{routine}.trc"
            pkl_path = f"/home/lea/trampo/4DHumans/outputs/results/demo_{routine}-{camera}.pkl"

            trc_data, marker_names = load_trc(trc_path)
            trc_positions = extract_trc_positions(trc_data, marker_names)
            frames = get_trc_frames(trc_data, marker_names)

            pkl_positions = load_3d_keypoints(pkl_path)

            A = trc_positions
            B = pkl_positions[frames]

            ### Find matching joints and reorder skeletons
            A = A[:, matching_pose2sim]
            B = B[:, :, matching_humans]

            ### Transform to align skeletons
            F, K_A, _ = A.shape        # (F, 13, 3)
            num_dets, K_B = B.shape[1], B.shape[2]  # (F, 2, 45, 3)

            assert K_A == K_B

            def compute_error(A, B, K_A, K_B):
                """ cost_matrix = np.zeros((K_A, K_B))
                for i in range(K_A):
                    for j in range(K_B):
                        cost_matrix[i, j] = np.linalg.norm(A[i] - B[j])

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                B_selected = B[col_ind]  # (13, 3) """

                error = np.nanmean(np.linalg.norm(A - B, axis=1))

                return error, B

            mpjpe_raw_all, mpjpe_pa_all, mpjpe_ka_all = [], [], []
            B_selected_all, B_selected_PA_all, B_selected_KA_all = [], [], []
            frame_indices = []
            PA_transfo, KA_transfo = [], []

            for t in range(F):
                A_frame = A[t]            # (13, 3)
                B_frame = B[t]            # (2, 45, 3)

                errors, PA_errors, KA_errors = [], [], []
                B_selected_list, B_selected_PA_list, B_selected_KA_list = [], [], []

                for person_idx in range(num_dets):
                    B_person = B_frame[person_idx]  # (45, 3)

                    if np.isnan(B_person).any() or np.isnan(A_frame).any():
                        continue

                    # Step 1: Compute assignment cost *before* alignment
                    error, B_selected = compute_error(A_frame, B_person, K_A, K_B)
                    errors.append(error)
                    B_selected_list.append(B_selected)

                    # Step 2.1: Align matching B keypoints (45 -> 13 joints) to A_frame (13 joints) with Procrustes
                    R, scale, t_vec = procrustes_align(A_frame, B_person[:15])  # or use Kabsch
                    B_person_PA = (B_person @ R) * scale + t_vec  # shape: (45, 3)
                    PA_transfo.append((R, scale, t_vec))

                    # Step 2.2: Align with Kabsch
                    R, t_vec = kabsch_align(A_frame, B_person[:15])
                    B_person_KA = (B_person @ R) + t_vec
                    KA_transfo.append((R, t_vec))

                    # Step 3: Compute assignment cost *after* alignment
                    PA_error, B_selected_PA = compute_error(A_frame, B_person_PA, K_A, K_B)
                    PA_errors.append(PA_error)
                    B_selected_PA_list.append(B_selected_PA)

                    KA_error, B_selected_KA = compute_error(A_frame, B_person_KA, K_A, K_B)
                    KA_errors.append(KA_error)
                    B_selected_KA_list.append(B_selected_KA)

                if len(errors) > 0: # at least 1 detection in frame
                    # Step 4: Find detection with lowest error *before* alignment
                    best_idx = np.argmin(errors)
                    best_B_selected = B_selected_list[best_idx]

                    # Step 5: Find detection with lowest error *after* alignment
                    best_idx_PA = np.argmin(PA_errors)
                    best_B_selected_PA = B_selected_PA_list[best_idx_PA]

                    best_idx_KA = np.argmin(KA_errors)
                    best_B_selected_KA = B_selected_KA_list[best_idx_KA]

                    B_selected_all.append(best_B_selected)                # (13, 3)
                    B_selected_PA_all.append(best_B_selected_PA)          # (13, 3)
                    B_selected_KA_all.append(best_B_selected_KA)          # (13, 3)
                    frame_indices.append(int(t))
                    
                    # Step 6: Compute mean errors
                    mpjpe_raw = np.nanmean(np.linalg.norm(A_frame - best_B_selected, axis=1))
                    mpjpe_pa = np.nanmean(np.linalg.norm(A_frame - best_B_selected_PA, axis=1))
                    mpjpe_ka = np.nanmean(np.linalg.norm(A_frame - best_B_selected_KA, axis=1))

                    mpjpe_raw_all.append(mpjpe_raw)
                    mpjpe_pa_all.append(mpjpe_pa)
                    mpjpe_ka_all.append(mpjpe_ka)

            B_selected_all = np.array(B_selected_all)             # shape (F_valid, 13, 3)
            B_selected_PA_all = np.array(B_selected_PA_all)
            B_selected_KA_all = np.array(B_selected_KA_all)

            A_valid = A[frame_indices]  # truncate A if needed (to match valid frames)

            # === Global Alignment ===
            R_global, t_global = kabsch_align(A_valid.reshape(-1, 3), B_selected_all.reshape(-1, 3))

            #print('Global rotation (x,y,z degrees):', rot.from_matrix(R_global).as_euler('xyz', degrees=True))  # shape (F, 3))
            #print('Global translation (m):', t_global)

            B_selected_global_all = (B_selected_all @ R_global) + t_global  # (F_valid, 13, 3)
            mpjpe_global_all = np.mean(np.linalg.norm(A_valid - B_selected_global_all, axis=2), axis=1)

            # #print results
            #print(f'Average MPJPE raw: {np.mean(mpjpe_raw_all):.3f} m')
            #print(f'Min MPJPE raw: {np.min(mpjpe_raw_all):.3f} m')
            #print(f'Max MPJPE raw: {np.max(mpjpe_raw_all):.3f} m')
            mpjpe_raw_dict[routine][camera].append(mpjpe_raw_all)

            #print(f'Average Global KA-MPJPE: {np.mean(mpjpe_global_all):.3f} m')
            #print(f'Min Global KA-MPJPE: {np.min(mpjpe_global_all):.3f} m')
            #print(f'Max Global KA-MPJPE: {np.max(mpjpe_global_all):.3f} m')
            mpjpe_global_dict[routine][camera].append(mpjpe_global_all.tolist())

            #print(f'Average PA-MPJPE:  {np.mean(mpjpe_pa_all):.3f} m')
            #print(f'Min PA-MPJPE:  {np.min(mpjpe_pa_all):.3f} m')
            #print(f'Max PA-MPJPE:  {np.max(mpjpe_pa_all):.3f} m')
            mpjpe_pa_dict[routine][camera].append(mpjpe_pa_all)

            #print(f'Average KA-MPJPE:  {np.mean(mpjpe_ka_all):.3f} m')
            #print(f'Min KA-MPJPE:  {np.min(mpjpe_ka_all):.3f} m')
            #print(f'Max KA-MPJPE:  {np.max(mpjpe_ka_all):.3f} m')
            mpjpe_ka_dict[routine][camera].append(mpjpe_ka_all)

            """ ### Visualize transformations distributions
            R_all, scale_all, t_all = [], [], []

            for R, t in KA_transfo:
                R_all.append(R)
                t_all.append(t)

            R_all = np.array(R_all)
            t_all = np.array(t_all)

            euler_all = rot.from_matrix(R_all).as_euler('xyz', degrees=True)  # shape (F, 3)
            mean_rot = np.mean(euler_all, axis=0)
            #print(mean_rot)
            #print(np.mean(t_all, axis=0))

            R_all_reconstructed = rot.from_euler('xyz', mean_rot, degrees=True).as_matrix()
            #print(R_all_reconstructed)

            plt.figure(figsize=(10, 3))
            for i, axis in enumerate(['X', 'Y', 'Z']):
                plt.subplot(1, 3, i+1)
                plt.hist(t_all[:, i], bins=30, color='salmon', edgecolor='k')
                plt.title(f'{axis}-Translation')
                plt.xlabel('Meters')
                plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'metrics/3dregistrationFigs/Translation_hist_{routine}_{camera}.png')
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 3))
            for i, axis in enumerate(['X', 'Y', 'Z']):
                plt.plot(t_all[:, i], label=f'{axis}-trans')
            plt.legend()
            plt.xlabel('Frame')
            plt.ylabel('Translation (m)')
            plt.title('Translation over Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'metrics/3dregistrationFigs/Translation_time_{routine}_{camera}.png')
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 3))
            for i, axis in enumerate(['X', 'Y', 'Z']):
                plt.subplot(1, 3, i+1)
                plt.hist(euler_all[:, i], bins=30, color='mediumseagreen', edgecolor='k')
                plt.title(f'{axis}-Rotation (deg)')
                plt.xlabel('Angle')
                plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'metrics/3dregistrationFigs/Rotation_hist_{routine}_{camera}.png')
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 3))
            for i, axis in enumerate(['X', 'Y', 'Z']):
                plt.plot(euler_all[:, i], label=f'{axis}-rot')
            plt.legend()
            plt.xlabel('Frame')
            plt.ylabel('Rotation (deg)')
            plt.title('Euler Angles over Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'metrics/3dregistrationFigs/Rotation_time_{routine}_{camera}.png')
            #plt.show()
            plt.close()

            ### Visualize MPJPE distribution
            frames = np.arange(len(mpjpe_raw_all))

            plt.figure(figsize=(14, 5))

            # --- MPJPE raw histogram ---
            all_data = np.concatenate([
                mpjpe_raw_all,
                mpjpe_global_all,
                mpjpe_pa_all,
                mpjpe_ka_all
            ])
            bin_edges = np.histogram_bin_edges(all_data, bins=40)

            plt.subplot(1, 3, 1)
            plt.hist(mpjpe_raw_all, bins=bin_edges, alpha=0.7, label='MPJPE raw', color='orange')
            plt.hist(mpjpe_global_all, bins=bin_edges, alpha=0.6, label='MPJPE KA global', color='red')
            plt.hist(mpjpe_pa_all, bins=bin_edges, alpha=0.7, label='PA-MPJPE', color='green')
            plt.hist(mpjpe_ka_all, bins=bin_edges, alpha=0.5, label='KA-MPJPE', color='blue')
            plt.xlabel('Error (m)')
            plt.ylabel('Frame count')
            plt.title('MPJPE Distribution')
            plt.legend()

            # --- MPJPE over time ---
            plt.subplot(1, 3, 2)
            plt.plot(frames, mpjpe_raw_all, label='MPJPE raw', color='orange')
            plt.plot(frames, mpjpe_global_all, label='MPJPE KA global', color='red')
            plt.plot(frames, mpjpe_pa_all, label='PA-MPJPE', color='green')
            plt.plot(frames, mpjpe_ka_all, label='KA-MPJPE', color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Error (m)')
            plt.title('MPJPE Over Time')
            plt.legend()

            # --- Boxplot ---
            plt.subplot(1, 3, 3)
            plt.boxplot([mpjpe_raw_all, mpjpe_global_all, mpjpe_pa_all, mpjpe_ka_all], labels=['MPJPE', 'MPJPE KA global', 'PA-MPJPE', 'KA-MPJPE'])
            plt.ylabel('Error (m)')
            plt.title('MPJPE Spread (Boxplot)')

            plt.tight_layout()
            plt.savefig(f'metrics/3dregistrationFigs/MPJPE_{routine}_{camera}.png')
            #plt.show()
            plt.close() """

            plot_joint_coordinates_over_time(A_valid, B_selected_all, B_selected_global_all, B_selected_KA_all)

    with open("mpjpe_raw_dict.json", "w") as outfile:
        json.dump(mpjpe_raw_dict, outfile)

    with open("mpjpe_global_dict.json", "w") as outfile:
        json.dump(mpjpe_global_dict, outfile)

    with open("mpjpe_pa_dict.json", "w") as outfile:
        json.dump(mpjpe_pa_dict, outfile)

    with open("mpjpe_ka_dict.json", "w") as outfile:
        json.dump(mpjpe_ka_dict, outfile)

import numpy as np
import pandas as pd

import os
import json
import itertools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Patch


def expand_header(header_line):
    new_header = []
    i = 0
    j = 0
    while i < len(header_line):
        entry = header_line[i]
        if entry not in ['', None]:
            if entry in ['Frame#', 'Time']:
                new_header.append(entry)
                i += 1
            else:
                new_header.extend([f"X{j}_{entry}", f"Y{j}_{entry}", f"Z{j}_{entry}"])
                j += 1
                i += 3  # Skip the next 2 empty strings
        else:
            i += 1  # Just skip if it's empty
    return new_header

def extract_coordinates(filename):
    # -- Charger le fichier TRC en sautant les 3 premières lignes --
    with open(filename, "r") as f:
        lines = f.readlines()

    # Extraire les noms de colonnes depuis la 4e ligne (index 3)
    column_names = lines[3].strip().split('\t')
    column_names = expand_header(column_names)

    # Charger le reste des données en DataFrame
    df = pd.read_csv(filename, 
                    sep='\t', 
                    skiprows=5, 
                    names=column_names)

    # -- Nettoyer les colonnes (en cas de colonnes vides) --
    df = df.dropna(axis=1, how='all')  # Supprimer les colonnes complètement vides

    # Extraire les noms des marqueurs
    marker_names, marker_indices = [], []
    for name in column_names[2:]:  # Ignorer 'Frame#' et 'Time'
        if name.startswith('X'):
            marker_indices.append(name[1:].split('_')[0])
            marker_names.append(name.split('_')[-1])

    # -- Convertir en numpy array (frames, keypoints, 3) --
    num_frames = df.shape[0]
    num_markers = len(marker_names)
    coords = np.zeros((num_frames, num_markers, 3))

    for (i, marker_id), marker_name in zip(enumerate(marker_indices), marker_names):
        coords[:, i, 0] = df[f'X{marker_id}_{marker_name}'].values
        coords[:, i, 1] = df[f'Y{marker_id}_{marker_name}'].values
        coords[:, i, 2] = df[f'Z{marker_id}_{marker_name}'].values
    
    coords *= 1000 #convert to mm

    frame_numbers = df['Frame#'].values

    return coords, frame_numbers

def load_2d_keypoints(filename):
    # Load JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    if len(data["people"]) > 0:
        coords = []
        conf = []
        for i in range(len(data['people'])):
            keypoints = data["people"][i]["pose_keypoints_2d"]
            keypoints = np.array(keypoints).reshape((-1, 3))

            # Access x, y, and confidence separately if needed
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            confidences = keypoints[:, 2]

            coords.append((x_coords, y_coords))
            conf.append(confidences)

        return coords, conf
    
    return None, None

def project_points_to_camera(points_3d, K, T, R, im_size=(1920,1080)):
    """
    Projects 3D points to 2D image coordinates using camera intrinsics and extrinsics.

    Args:
        points_3d: (N, 3) array of 3D points.
        K: (3, 3) intrinsic matrix.
        T: (4, 4) extrinsic matrix (camera pose).
        im_size: tuple (width, height), optional. If provided, returns a mask for points inside the image.

    Returns:
        points_2d: (N, 2) projected 2D points.
        valid_mask: (N,) boolean array (only if im_size is given).
    """
    points_3d = np.array(points_3d) # shape (3, N)

    points_3d_r = (R @ points_3d.T).T

    points_3d_h = np.hstack((points_3d_r, np.ones((points_3d_r.shape[0], 1))))
    points_cam = T @ points_3d_h.T # shape (3, N)
    
    points_cam = points_cam[0:3,:] / points_cam[3,:]
    points_2d_h = K @ points_cam  # shape (3, N)

    points_2d = points_2d_h[:2, :] / points_2d_h[2, :]  # normalize
    points_2d = points_2d.T

    if im_size is not None:
        width, height = im_size
        x, y = points_2d[:, 0], points_2d[:, 1]
        valid_mask = (
            (points_cam[2, :] > 0) &  # In front of camera
            (x >= 0) & (x < width) &
            (y >= 0) & (y < height)
        )
        return points_2d, valid_mask

    return points_2d, np.ones(points_2d.shape)  # shape (N, 2)

def get_frame_name(model, frame_idx):
    if model == 'vit':
        frame_name = f'frame_{frame_idx:05d}_{frame_idx:06d}'
    elif model == '4dhumans':
        frame_name = f'frame_{frame_idx:06d}'
    return frame_name

def get_MPJPE_per_cam(model, cameras, R, K, projMat, matching_2d, matching_3d):
    if model == 'vit':
        source_root = '/home/lea/trampo/Pose2Sim/pose_all_vit'
        triang_root = '/home/lea/trampo/Pose2Sim/pose-3d-vit-multi'
        
    elif model == '4dhumans':
        source_root = '/home/lea/trampo/Pose2Sim/pose_all_4dhumans'
        triang_root = '/home/lea/trampo/Pose2Sim/pose-3d-4DHumans-multi'

    sequence_names = set(f.split('-')[0] for f in os.listdir(source_root))
    dist_per_cam = {}

    for cam_idx in range(8):
        cam_name = cameras[cam_idx]
        dist_per_cam.update({cam_name: {}})
        print(f'--- C{cam_idx+1} ---')
        
        for cam_combination in itertools.combinations(enumerate(cameras), 4):
            cam_indices, cam_names = zip(*cam_combination)
            if cam_idx in cam_indices:
                continue
            cam_label = '+'.join(f"C{idx+1}" for idx in cam_indices)  # Label like C1+C3+C5
            dist_per_cam[cam_name].update({cam_label:[]})
            dist = []
            
            # keypoints 2d cam i
            for seq in sorted(sequence_names):
                folder_name = f"{seq}-{cam_name}_json"
                src_path = os.path.join(source_root, folder_name)

                # keypoints 3d 4 cams
                triang_path = os.path.join(triang_root, seq, cam_label, f'{seq}.trc')
                try:
                    triang_points, frame_numbers = extract_coordinates(triang_path)
                except FileNotFoundError:
                    continue

                # frames loop
                for frame_idx in range(0, len(os.listdir(src_path)), 5):
                    # 2d
                    frame_name = get_frame_name(model, frame_idx)
                    try:
                        vit_pts, _ = load_2d_keypoints(f'{src_path}/{frame_name}.json')
                    except FileNotFoundError:
                        continue
                    
                    if vit_pts is not None:
                        vit_pts = np.array(vit_pts)
                        n_detections = vit_pts.shape[0]
                        vit_pts = vit_pts.reshape((n_detections, 2, -1))
                        vit_pts = vit_pts[:, :, matching_2d]
                    else:
                        continue
                    
                    # 3d
                    coords = triang_points[np.where(frame_numbers == frame_idx)].reshape((-1, 3))
                    if len(coords) > 0:
                        coords = coords[matching_3d]
                    else:
                        continue

                    # reproj 3d -> 2d cam i
                    reproj, _ = project_points_to_camera(coords, K[cam_idx], projMat[cam_idx], R)

                    # OPTIONAL : visualize detections and reprojection on images
                    """ fig, ax = plt.subplots(figsize=(19,10))
                    print(seq, frame_idx, cam_name)
                    img = mpimg.imread(f'/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Images_trampo_avril2025/20250429/{seq}-{cameras[cam_idx]}/frame_{frame_idx:05d}.png')
                    ax.imshow(img)
                    plt.scatter(*reproj.T, label='reprojection')
                    draw_skeleton(reproj, 'blue', ax, model)
                    for i in range(vit_pts.shape[0]):
                        plt.scatter(*vit_pts[i], label=f'detection {i+1}')
                        draw_skeleton(vit_pts[i].T, 'orange', ax, model)
                    #ax.axis('equal')
                    ax.set_xlim(0, 1920)
                    ax.set_ylim(0, 1080)
                    plt.legend()
                    plt.show() """
                    
                    # distance 2d
                    min_dist = np.full(reproj.shape[0], np.inf)
                    for j in range(vit_pts.shape[0]):
                        vit_reshaped = vit_pts[j].T
                        dist_j = np.linalg.norm(reproj - vit_reshaped, axis=1)
                        if np.mean(dist_j) < np.mean(min_dist):
                            min_dist = dist_j
                    dist.append(min_dist)
                
            dist_per_cam[cam_name][cam_label] = dist

    return dist_per_cam

#from compute_CoM import CoM
#com_obj = CoM('women', 13)

from filter_same_person import same_person
def get_MPJPE_per_cam_CoM(model, cameras, R, K, projMat, matching_2d, matching_3d, threshold_CoM=150, threshold_IoU=0.5):
    if model == 'vit':
        source_root = '/home/lea/trampo/Pose2Sim/pose_all_vit'
        triang_root = '/home/lea/trampo/Pose2Sim/pose-3d-vit-multi'
        
    elif model == '4dhumans':
        source_root = '/home/lea/trampo/Pose2Sim/pose_all_4dhumans'
        triang_root = '/home/lea/trampo/Pose2Sim/pose-3d-4DHumans-multi'

    sequence_names = set(f.split('-')[0] for f in os.listdir(source_root))
    dist_per_cam = {}

    for cam_idx in range(8):
        cam_name = cameras[cam_idx]
        dist_per_cam.update({cam_name: {}})
        print(f'--- C{cam_idx+1} ---')
        
        for cam_combination in itertools.combinations(enumerate(cameras), 4):
            cam_indices, cam_names = zip(*cam_combination)
            if cam_idx in cam_indices:
                continue
            cam_label = '+'.join(f"C{idx+1}" for idx in cam_indices)  # Label like C1+C3+C5
            dist_per_cam[cam_name].update({cam_label:[]})
            dist = []
            
            # keypoints 2d cam i
            for seq in sorted(sequence_names):
                folder_name = f"{seq}-{cam_name}_json"
                src_path = os.path.join(source_root, folder_name)

                # keypoints 3d 4 cams
                triang_path = os.path.join(triang_root, seq, cam_label, f'{seq}.trc')
                try:
                    triang_points, frame_numbers = extract_coordinates(triang_path)
                except FileNotFoundError:
                    continue

                # frames loop
                for frame_idx in range(0, len(os.listdir(src_path)), 10):
                    # 2d
                    frame_name = get_frame_name(model, frame_idx)
                    try:
                        vit_pts, _ = load_2d_keypoints(f'{src_path}/{frame_name}.json')
                    except FileNotFoundError:
                        continue

                    if vit_pts is not None:
                        vit_pts = np.array(vit_pts)
                        n_detections = vit_pts.shape[0]
                        vit_pts = vit_pts.reshape((n_detections, 2, -1))
                        vit_pts = vit_pts[:, :, matching_2d]
                    else:
                        continue
                    
                    # 3d
                    coords = triang_points[np.where(frame_numbers == frame_idx)].reshape((-1, 3))
                    if len(coords) > 0:
                        coords = coords[matching_3d]
                    else:
                        continue

                    # reproj 3d -> 2d cam i
                    reproj, _ = project_points_to_camera(coords, K[cam_idx], projMat[cam_idx], R)
                    
                    # distance 2d + filter on CoM position
                    
                    min_dist = np.full(reproj.shape[0], np.inf)
                    
                    for j in range(vit_pts.shape[0]):
                        vit_reshaped = vit_pts[j].T
                        dist_j = np.linalg.norm(reproj - vit_reshaped, axis=1)

                        """ com_reproj = com_obj.compute_global_cm(reproj)
                        com_detect = com_obj.compute_global_cm(vit_reshaped)

                        if np.linalg.norm(com_reproj - com_detect) < threshold_CoM and np.mean(dist_j) < np.mean(min_dist):
                            min_dist = dist_j """

                        if same_person(vit_reshaped, reproj, threshold_CoM, threshold_IoU) and np.mean(dist_j) < np.mean(min_dist):
                            min_dist = dist_j

                    if min_dist.sum() != np.inf:        
                        dist.append(min_dist)
                
            dist_per_cam[cam_name][cam_label] = dist

    return dist_per_cam

def plot_MPJPE_per_cam(MPJPE_per_cam, model, showNaN=False, ylim=False):
    n_best, n_worst = 4, 4

    for cam_label, dists in MPJPE_per_cam.items():
        labels = list(dists.keys())
        valeurs_list = list(dists.values())
        max_len = max(len(v) for v in valeurs_list)
        size = np.array(valeurs_list[0]).shape[-1]

        # Remplissage des valeurs avec NaN
        valeurs = np.full((len(valeurs_list), max_len * size), np.nan)
        for i, arr in enumerate(valeurs_list):
            r_arr = np.ravel(arr)
            valeurs[i, :len(r_arr)] = r_arr
        valeurs_sans_nan = [row[~np.isnan(row)] for row in valeurs]

        # Métriques
        nan_counts = np.isnan(valeurs).sum(axis=1)
        error_medians = np.nanmedian(valeurs, axis=1)
        error_q1 = np.nanquantile(valeurs, 0.25, axis=1)
        error_q3 = np.nanquantile(valeurs, 0.75, axis=1)
        error_maxiq = error_q3 + 1.5*(error_q3 - error_q1)

        if nan_counts.max() == 0:
            showNaN = False

        # --- Score combiné normalisé ---
        score_combined = error_medians / error_medians.max() + nan_counts / nan_counts.max() if showNaN else error_medians / error_medians.max()
        best_indices = np.argsort(score_combined)[:n_best]
        worst_indices = np.argsort(score_combined)[-n_worst:]

        # --- Création des axes ---
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Boxplot des erreurs (orange)
        bp = ax1.boxplot(valeurs_sans_nan, positions=np.arange(len(labels)),
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('tab:blue')
        ax1.set_ylabel('Erreur 2D-MPJPE (px)', color='tab:blue', fontweight='bold')

        if not ylim:
            ax1.set_ylim(0, max(error_maxiq) * 1.05)
        else:
            ax1.set_ylim(0, ylim)

        if showNaN:
            ax2 = ax1.twinx()
            # Barres du nombre de NaN (bleu)
            bars = ax2.bar(np.arange(len(labels)), nan_counts, alpha=0.4,
                        color='skyblue', width=0.6)
            ax2.set_ylabel('Nombre total de points manquants (NaN)', color='skyblue', fontweight='bold')
            ax2.set_ylim(0, max(nan_counts) * 1.05)

            # --- Légende partagée ---
            legend_elements = [Patch(facecolor='tab:blue', label='Erreur 2D-MPJPE (px)'),
                            Patch(facecolor='skyblue', alpha=0.4, label='Nombre total de NaN')]
            ax1.legend(handles=legend_elements, loc='upper left')

        # --- Coloration des labels ---
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels, rotation=90)
        xticklabels = ax1.get_xticklabels()
        for idx in worst_indices:
            xticklabels[idx].set_color('red')
            xticklabels[idx].set_fontweight('bold')
        for idx in best_indices:
            xticklabels[idx].set_color('green')
            xticklabels[idx].set_fontweight('bold')

        # --- Grilles et titre ---
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_title(f'Erreur et NaN post triangulation sur la {cam_label}')

        plt.tight_layout()
        plt.savefig(f'/home/lea/trampo/metrics/Pose2Sim_triangulation_figs/{model}_reproj_NaN__filtCoM_{cam_label}.png')
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_MPJPE_all_cams(MPJPE_per_cam, model, showNaN=False, ylim=False):
    """
    Fait un seul graphique avec les caméras sur l'axe horizontal
    et les distributions d'erreurs regroupées par caméra.
    """

    n_best, n_worst = 4, 4

    cam_labels = list(MPJPE_per_cam.keys())
    valeurs_par_cam = []
    nan_props = []
    error_medians = []
    error_q1 = []
    error_q3 = []
    error_maxiq = []

    for cam_label, dists in MPJPE_per_cam.items():
        valeurs_list = list(dists.values())
        max_len = max(len(v) for v in valeurs_list)
        size = np.array(valeurs_list[0]).shape[-1]

        # Remplissage avec NaN
        valeurs = np.full((len(valeurs_list), max_len * size), np.nan)
        for i, arr in enumerate(valeurs_list):
            r_arr = np.ravel(arr)
            valeurs[i, :len(r_arr)] = r_arr
        valeurs_sans_nan = [row[~np.isnan(row)] for row in valeurs]

        # Sauvegarde pour ce cam
        valeurs_par_cam.append(np.concatenate(valeurs_sans_nan) if len(valeurs_sans_nan) > 0 else np.array([]))

        n_total = np.prod(valeurs.shape)
        n_nan = np.isnan(valeurs).sum()
        nan_props.append(n_nan / n_total if n_total > 0 else 0)

        error_medians.append(np.nanmedian(valeurs))
        q1 = np.nanquantile(valeurs, 0.25)
        q3 = np.nanquantile(valeurs, 0.75)
        error_q1.append(q1)
        error_q3.append(q3)
        error_maxiq.append(q3 + 1.5*(q3 - q1))

    nan_props = np.array(nan_props)
    error_medians = np.array(error_medians)
    error_q1 = np.array(error_q1)
    error_q3 = np.array(error_q3)
    error_maxiq = np.array(error_maxiq)

    # --- Score combiné ---
    score_combined = error_medians / error_medians.max()
    if showNaN and nan_props.max() > 0:
        score_combined += nan_props / nan_props.max()
    best_indices = np.argsort(score_combined)[:n_best]
    worst_indices = np.argsort(score_combined)[-n_worst:]

    # --- Création du plot ---
    fig, ax1 = plt.subplots(figsize=(15, 8))

    bp = ax1.boxplot(valeurs_par_cam, positions=np.arange(len(cam_labels)),
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('tab:blue')
    ax1.set_ylabel('Erreur 2D-MPJPE (px)', color='tab:blue', fontweight='bold')

    if not ylim:
        ax1.set_ylim(0, max(error_maxiq) * 1.05)
    else:
        ax1.set_ylim(0, ylim)

    if showNaN:
        ax2 = ax1.twinx()
        bars = ax2.bar(np.arange(len(cam_labels)), nan_props, alpha=0.4,
                       color='skyblue', width=0.6)
        ax2.set_ylabel('Proportion de NaN', color='skyblue', fontweight='bold')
        ax2.set_ylim(0, 1.05)  # entre 0 et 100 %
        ax2.set_yticks(np.linspace(0, 1, 6))
        ax2.set_yticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 6)])

        legend_elements = [Patch(facecolor='tab:blue', label='Erreur 2D-MPJPE (px)'),
                           Patch(facecolor='skyblue', alpha=0.4, label='Proportion de NaN')]
        ax1.legend(handles=legend_elements, loc='upper left')

    # --- Labels colorés ---
    ax1.set_xticks(np.arange(len(cam_labels)))
    ax1.set_xticklabels(cam_labels, rotation=45)
    xticklabels = ax1.get_xticklabels()
    for idx in worst_indices:
        xticklabels[idx].set_color('red')
        xticklabels[idx].set_fontweight('bold')
    for idx in best_indices:
        xticklabels[idx].set_color('green')

        xticklabels[idx].set_fontweight('bold')

    # --- Grille et titre ---
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_title(f'Erreur de reprojection et proportion de NaN post triangulation – modèle {model}')

    plt.tight_layout()
    plt.savefig(f'/home/lea/trampo/metrics/Pose2Sim_triangulation_figs/{model}_reproj_NaN__filtCoM_ALL.png')
    plt.show()


def get_stats(values):
    arr = np.array(values)
    return {
        "moyenne": np.mean(arr),
        "médiane": np.median(arr),
        "écart_type": np.std(arr, ddof=1),  # écart-type échantillon
        "q1": np.percentile(arr, 25),
        "q3": np.percentile(arr, 75),
        "max": np.min(arr),
        "max": np.max(arr)
    }

def get_all_error_stats(data):
    # --- 1) Statistiques par combinaison ---
    stats_par_combinaison = {cam: {comb: get_stats(vals) for comb, vals in comb_dict.items()}
                            for cam, comb_dict in data.items()}
    # --- 2) Statistiques par caméra ---
    stats_par_camera = {}
    for cam, comb_dict in data.items():
        toutes_valeurs = np.concatenate(list(comb_dict.values()))
        stats_par_camera[cam] = get_stats(toutes_valeurs)

    # --- 3) Statistiques globales ---
    toutes_valeurs_global = np.concatenate([vals for comb_dict in data.values() for vals in comb_dict.values()])
    stats_globales = get_stats(toutes_valeurs_global)

    # Affichage
    print("\n--- Stats par combinaison ---")
    for cam, combs in stats_par_combinaison.items():
        print(cam)
        for comb, stats in combs.items():
            print(f"  {comb}: {stats}")

    print("\n--- Stats par caméra ---")
    for cam, stats in stats_par_camera.items():
        print(f"{cam}: {stats}")

    print("\n--- Stats globales ---")
    print(stats_globales)
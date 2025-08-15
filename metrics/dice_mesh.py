import os
import trimesh
import numpy as np
import joblib

import torch
import smplx

import tempfile
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

# --- Create meshes ---
def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert a batch of rotation matrices (N, 3, 3) to axis-angle (N, 3).
    """
    def _angle_axis_from_rotmat(R):
        cos_theta = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        rx = R[..., 2, 1] - R[..., 1, 2]
        ry = R[..., 0, 2] - R[..., 2, 0]
        rz = R[..., 1, 0] - R[..., 0, 1]
        axis = torch.stack([rx, ry, rz], dim=-1)

        axis = axis / (2 * torch.sin(theta).unsqueeze(-1) + 1e-8)
        axis_angle = axis * theta.unsqueeze(-1)

        axis_angle[torch.isnan(axis_angle)] = 0.0
        return axis_angle

    return _angle_axis_from_rotmat(rotation_matrix)

def save_meshes(path, routine, cameras, smpl_model):
    for cam in cameras:
        results_path = path + f'/demo_{routine}-{cam}.pkl'
        results = joblib.load(results_path)

        for frame_idx in range(1, len(results)):
            frame = f'outputs//_DEMO/{routine}-{cam}/img/{frame_idx:06d}.jpg'

            detections = results[frame]['smpl']

            all_vertices = []
            all_faces = []
            offset = 0

            for id, output in enumerate(detections):
                pose = torch.tensor(output['body_pose'], dtype=torch.float32).unsqueeze(0)         # (1, 23, 3, 3)
                betas = torch.tensor(output['betas'], dtype=torch.float32).unsqueeze(0)             # (1, 10)
                global_orient = torch.tensor(output['global_orient'], dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
                transl = np.array(results[frame]['pose'][id][-3:], dtype=np.float32).reshape(1,3)

                body_pose_aa = rotation_matrix_to_angle_axis(pose).reshape(1, -1)        # (1, 69)
                global_orient_aa = rotation_matrix_to_angle_axis(global_orient).reshape(1, 3)  # (1, 3)

                output_smpl = smpl_model(
                    betas=betas,
                    body_pose=body_pose_aa,
                    return_verts=True
                )

                vertices = output_smpl.vertices[0].detach().cpu().numpy()

                """ # Apply translation if available
                if transl is not None:
                    vertices += transl """

                faces = smpl_model.faces + offset  # Shift face indices
                offset += vertices.shape[0]

                all_vertices.append(vertices)
                all_faces.append(faces)

            # Combine all into one mesh
            if all_vertices:
                all_vertices = np.vstack(all_vertices)
                all_faces = np.vstack(all_faces)

                combined_mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)

                save_folder = f"meshes/{routine}/{cam}"
                os.makedirs(save_folder, exist_ok=True)

                save_path = f"meshes/{routine}/{cam}/{frame_idx:06d}.ply"
                combined_mesh.export(save_path)
    
    return

def create_mesh(path, routine, camera, frame_idx, smpl_model):
    results_path = path + f'/demo_{routine}-{camera}.pkl'
    results = joblib.load(results_path)

    frame = f'outputs//_DEMO/{routine}-{camera}/img/{frame_idx:06d}.jpg'

    try:
        detections = results[frame]['smpl']
    except KeyError:
        return None

    all_vertices = []
    all_faces = []
    offset = 0

    for id, output in enumerate(detections):
        pose = torch.tensor(output['body_pose'], dtype=torch.float32).unsqueeze(0)         # (1, 23, 3, 3)
        betas = torch.tensor(output['betas'], dtype=torch.float32).unsqueeze(0)             # (1, 10)
        global_orient = torch.tensor(output['global_orient'], dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
        transl = np.array(results[frame]['pose'][id][-3:], dtype=np.float32).reshape(1,3)

        body_pose_aa = rotation_matrix_to_angle_axis(pose).reshape(1, -1)        # (1, 69)
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient).reshape(1, 3)  # (1, 3)

        output_smpl = smpl_model(
            betas=betas,
            body_pose=body_pose_aa,
            return_verts=True
        )

        vertices = output_smpl.vertices[0].detach().cpu().numpy()

        """ # Apply translation if available
        if transl is not None:
            vertices += transl """

        faces = smpl_model.faces + offset  # Shift face indices
        offset += vertices.shape[0]

        all_vertices.append(vertices)
        all_faces.append(faces)

    # Combine all into one mesh
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)

        combined_mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)
    
        return combined_mesh
    
    return None

# --- Process meshes ---
def aligned_voxelization_full(meshes, pitch):
    """
    Voxelise plusieurs meshes dans un espace commun et retourne :
    - les matrices voxelisées alignées (0/1)
    - les voxel_meshes (as_boxes) alignés dans le monde réel
    """
    # 1. Trouver les bornes globales
    bounds = np.array([m.bounds for m in meshes])
    global_min = bounds[:, 0, :].min(axis=0)
    global_max = bounds[:, 1, :].max(axis=0)
    dims = np.ceil((global_max - global_min) / pitch).astype(int)
    shape = tuple(dims)

    voxel_matrices = []

    for mesh in meshes:
        # 2. Décaler dans le repère commun
        shifted = mesh.copy()
        shifted.apply_translation(-global_min)

        # 3. Voxelisation
        vox = shifted.voxelized(pitch).fill()
        mat = np.zeros(shape, dtype=np.uint8)

        indices = np.floor(vox.points / pitch).astype(int)
        for idx in indices:
            if np.all((0 <= idx) & (idx < dims)):
                mat[tuple(idx)] = 1

        voxel_matrices.append(mat)

    return voxel_matrices

# --- Compute metrics ---
def compute_dice(mats):
    intersection = np.sum(mats[0] & mats[1])
    total = np.sum(mats[0]) + np.sum(mats[1])
    dice = 2.0 * intersection / total if total > 0 else -1.0
    return dice

def build_dice_matrix(meshes1, meshes2, pitch):
    N = len(meshes1)
    dice_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            voxel_matrices, _ = aligned_voxelization_full([meshes1[i], meshes2[j]], pitch)
            dice = compute_dice([voxel_matrices[0], voxel_matrices[1]])
            dice_matrix[i, j] = dice
    return dice_matrix


# --- Plot results ---
def plot_with_pyvista(meshes, voxel_matrices, pitch):
    plotter = pv.Plotter()

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10)[:3] for i in range(len(meshes)+1)]

    # --- Affichage des meshes originaux ---
    for i, mesh in enumerate(meshes):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            mesh.export(tmp.name, file_type='ply')
            pv_mesh = pv.read(tmp.name)
            color_rgb = [int(255 * c) for c in colors[i]]
            plotter.add_mesh(pv_mesh, color=color_rgb, opacity=0.4)
            os.unlink(tmp.name)

    # --- Points de l'intersection des voxel_matrices ---
    intersection = voxel_matrices[0] & voxel_matrices[1]
    color = colors[2]
    
    indices = np.argwhere(intersection == 1)
    if len(indices) > 50000:
        indices = indices[np.random.choice(len(indices), 50000, replace=False)]
    points = indices * pitch
    cloud = pv.PolyData(points)
    color_rgb = [int(255 * c) for c in color]
    plotter.add_mesh(cloud, color=color_rgb, point_size=5, render_points_as_spheres=True)

    plotter.show()


if __name__ == "__main__":

    # Save meshes
    path = '/home/lea/trampo/4DHumans/outputs/results'

    video_path = '/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Images_trampo_avril2025/20250429/'
    results_path = '/home/lea/trampo/metrics/results/SMPL_keypoints'

    pitch = 0.01

    routines = sorted([os.path.basename(f).split('-')[0] for f in os.listdir(results_path)])
    routines = set(routines)

    cameras = ['Camera1_M11139', 'Camera2_M11140', 'Camera3_M11141', 'Camera4_M11458',
               'Camera5_M11459', 'Camera6_M11461', 'Camera7_M11462', 'Camera8_M11463']
    
    # Load SMPL model
    model = smplx.create(model_path='models/smpl', model_type='smpl', gender='neutral', batch_size=1)

    for routine in routines:
        N_frames = len(os.listdir(video_path+'/'+routine+'-Camera1_M11139'))
        print(routine, 'N_frames', N_frames)

        for frame_idx in range(1, N_frames, 100):
            for cam1 in cameras:
                combined_meshes1 = create_mesh(path, routine, cam1, frame_idx, model)

                if combined_meshes1 is not None:
                    meshes1 = combined_meshes1.split()
                else:
                    continue

                for cam2 in cameras[cameras.index(cam1)+1:]:
                    combined_meshes2 = create_mesh(path, routine, cam2, frame_idx, model)

                    if combined_meshes2 is not None:
                        meshes2 = combined_meshes2.split()
                    else:
                        continue

                    """ for i, mesh1 in enumerate(meshes1):
                        best_dice = 0
                        for j, mesh2 in enumerate(meshes2):

                            mats = aligned_voxelization_full([mesh1, mesh2], pitch)
                            dice = compute_dice(mats)

                            if dice > best_dice:
                                best_dice = dice """
                    
                    dice_mat = build_dice_matrix(meshes1, meshes2, pitch)
                    row_ind, col_ind = linear_sum_assignment(-dice_mat)  # Maximisation

                    #print(dice_mat)
                    for i, j in zip(row_ind, col_ind):
                        print(f"{frame_idx:06d}   {len(meshes1)} - {len(meshes2)}   Best match: mesh1[{i}] ↔ mesh2[{j}]  Dice = {dice_mat[i,j]:.4f}")
                        
                    break
                break
            #break
        #break

